import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import datetime
import pkg_resources

from pytorch_lightning.callbacks import Callback
from cycler import cycler
from enum import Enum

from tud_presence_prediction.helpers.profiling.time_profiling import TimeProfiler

class logger(Callback):
    """
    Adds automatic logging based on the lightning models callbacks, if given to the lightning trainer. 
    - Automatically adds text logs for some metrics
    - Automatically writes the experiments last metrics to the overview file if they have been provided
    - Automatically updates graph if start_auto_graph() was called
    - Automatically uses TimeProfiler if set up

    Additionally offers methods for manually logging text to the relevant version directory with some automatic formatting.
    - info() for general information
    - headline() for headlines
    - critical() for very important messages

    Additionally offers general graph functionality:
    - Function to stylize any graph to fit the general theme (plot line colors must be set manually)
    - Function to plot any users presence
    """

    use_style = True
    image_name = "visual_metrics{name_addition}.png"
    text_name = "text_log.txt"
    version_overview_name = "version_metrics.csv"
    base_results_path = "training_results"
    default_metrics = ["Training loss", "Training precision", "Training accuracy", "Validation loss", "Validation precision", "Validation accuracy"]

    # visual settings for generated graphs
    beige_colors = ["#FFFFFF", "#eae5de", "#d0c7b6"]
    blue_colors = ["#F8F9FA", "#DADCE0", "#BDC1C6", "#9AA0A6"] 
    pink_colors = ["#f3dfee", "#f2bec6", "#e4a38c"]
    variety_colors = ["#D2E6F0", "#d0c7b6", "#FFFFFF", "#000000", "#e4a38c"] # alt. beige: #eae5de, #F7EEDF  #alt blue #C8F2FF, #D2E6F0
    line_styles = ["solid", (0, (1, 0.5)), (0, (0.5, 0.1)), (0, (5, 1)), (0, (1, 10))]
    marker_styles = ["v", "d", "o", "1", "8"]
    marker_size = 4
    line_widths = [1.5, 0.5, 0.5]

    def __init__(self, mode, logging_directory=None, experiment_version=None, use_time_profiler=False):
        self.logging_directory = logging_directory
        self.current_experiment_version = experiment_version
        self.mode = logger_mode[mode.upper()] if isinstance(mode, str) else mode

        self.full_log = ""

        self.current_plot_figure = None
        self.current_plot_ax = None
        self.current_plot_lines = dict()
        self.current_plot_dataloader = None
        self.current_annotations = []
        self.current_decorations = []
        self.extra_sanity_data = dict()
        self.max_epochs = None
        self.live_show_graph = False
        self.live_save_graph = False
        self.horizontal_grid_steps = None
        self.user_number = None
        self.starting_epoch = 0
        self.use_time_profiler = use_time_profiler

        TimeProfiler.active = use_time_profiler
    
    def critical(self, text):
        if self.mode <= logger_mode.NONE: return

        self._console_log(text.upper())
        self._file_log(text.upper())
    
    def info(self, text, new_line=False):
        if self.mode <= logger_mode.NONE: return

        self._console_log(text, new_line)
        self._file_log(text, new_line)

    def headline(self, text):
        if self.mode <= logger_mode.NONE: return

        log_text = "---" + text + "---"
        self._console_log(log_text, True)
        self._file_log(log_text, True)

    def start_training_graph(self, lightning_model, max_epochs, data_loader, show_graph, save_graph, cloud=False):
        if self.mode < logger_mode.FULL: return
        if self.current_plot_figure != None: raise NotImplemented("Only one graph at a time supported for now. ")

        if not plt.isinteractive: plt.ion()
        if cloud:
            matplotlib.use('Agg')
        else:
            matplotlib.use('TkAgg')

        # calculate step size for horizontal grid lines
        max_points = max_epochs
        self.horizontal_grid_steps = round(max_points/float(20))
        if self.horizontal_grid_steps < 2: self.horizontal_grid_steps = 2
        for additive in range(10):
            if max_points % (self.horizontal_grid_steps + additive) == 0: 
                self.horizontal_grid_steps += additive
                break

        # configure graph
        #plt.rcParams["axes.prop_cycle"] = cycler('color', ["#FFFFFF", "#D9ECF2", "#FFDEAD", "#FFC0B4", "#86b7cf", "#90A180", "#C9B0E7", "#00BB2D", "#587246", "#354D73", "#5E2129", "#EC7C26"])
        #plt.rcParams["axes.prop_cycle"] = cycler('color', ["#FFFFFF", "#f7f5f2", "#eae5de", "#ddd6ca", "#d0c7b6"])
        self.current_plot_figure = plt.figure(figsize=(14, 6))
        self.current_plot_dataloader = data_loader
        self.current_plot_ax = self.current_plot_figure.add_subplot(111, autoscale_on=True, ylim=(-0.05,1.05), xlim=(self.starting_epoch-1, self.starting_epoch + max_epochs))
        self.current_plot_ax.xaxis.set_major_locator(plt.MultipleLocator(self.horizontal_grid_steps))
        self.current_plot_ax.set_xlabel("Epochs")
        self.current_plot_ax.set_ylabel("Loss")
        self.current_plot_ax.set_title(self.get_experiment_string(), color="white", fontweight="bold", y=1.04)

        # style graph
        self._set_plot_style()

        self.max_epochs = max_epochs

        self.live_show_graph = show_graph
        self.live_save_graph = save_graph
        self._update_training_graph(lightning_model)

    def save_training_graph(self):
        if self.mode < logger_mode.FULL: return
        if self.current_plot_figure == None: raise ModuleNotFoundError("Can not save figure before creating it.")

        additional_filename = "_user" + str(self.user_number) if self.user_number != None else ""
        self.current_plot_figure.savefig(self.logging_directory + os.path.sep + self.image_name.format(name_addition = additional_filename), bbox_inches='tight')
    
    def show_training_graph_permanently(self):
        if self.live_show_graph: plt.pause(100000)

    def generate_user_presence_graph(self, user_presences_baseline, user_presences_1=None, user_presences_2=None, title=None, timeslots_per_day=52):
        import matplotlib.patheffects as pe

        self._set_plot_style()
        plt.figure()
        temp_ax = plt.gca()
        temp_fig = plt.gcf()
        self._set_plot_style(temp_ax, temp_fig)

        if title != None: plt.title(title)

        baseline_presence_plot = plt.plot(user_presences_baseline, label="Real presence", lw=4, color="#000000", zorder=1, path_effects=[pe.Stroke(linewidth=5, foreground='#ffffff'), pe.Normal()])[0]
        self._decorate_line(baseline_presence_plot, temp_ax, 1.5, "#ffffff")
        
        if user_presences_1 is not None: 
            user_presence_plot_1 = plt.plot(user_presences_1, label="Predicted presence", lw=2, color="#ffffff", zorder=10)[0]
            self._decorate_line(user_presence_plot_1, temp_ax, 0.3, "#ffffff")
        if user_presences_2 is not None: 
            user_presence_plot_2 = plt.plot(user_presences_2, label="Other Presence", lw=1, color=self.variety_colors[1], zorder=20)[0]
            self._decorate_line(user_presence_plot_2, temp_ax, 0)

        
            
        temp_ax.xaxis.set_major_locator(plt.MultipleLocator(timeslots_per_day*6))
        temp_ax.xaxis.set_minor_locator(plt.MultipleLocator(timeslots_per_day))
        temp_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x_val, x_pos: int(x_val/timeslots_per_day)))
        temp_ax.xaxis.set_label_text(f"Day")
        temp_ax.set_ylim(-0.3, 1.3)
        temp_ax.yaxis.set_ticks([0, 1])
        temp_ax.yaxis.set_ticklabels(["Absent", "Present"])
        
        temp_ax.grid(which='minor', alpha=0.1)
        temp_fig.set_size_inches(15, 7)
        plt.legend(loc="upper right") 

        

    def _update_training_graph(self, lightning_model):
        if not lightning_model or not lightning_model.visual_metrics: return
        if self.mode < logger_mode.FULL: return

        #self.current_plot_ax.clear()

        self._remove_decorations()
        self._remove_annotations()

        last_metric_count = len(self.current_plot_lines)

        train_metric_count = 0
        val_metric_count = 0
        unknown_type_count = 0
        for graph_index, graph_metric in enumerate(lightning_model.visual_metrics):

            metric_set_type = 0 if "training" in str(graph_metric).lower() else (1 if "validation" in str(graph_metric).lower() else 2)
            metric_type = 0 if "precision" in str(graph_metric).lower() else (1 if "accuracy" in str(graph_metric).lower() else (2 if "loss" in str(graph_metric) else (3 if "LR" in str(graph_metric) else 4 + unknown_type_count)))
            
            metric_color = self.variety_colors[metric_type] #self.pink_colors[metric_type] if metric_set_type == 0 else (self.blue_colors[metric_type] if metric_set_type == 1 else "#000000")
            metric_line_style = self.line_styles[metric_set_type]
            metric_line_width = self.line_widths[metric_set_type]
            metric_marker = self.marker_styles[metric_type]

            if metric_set_type == 0: train_metric_count += 1
            elif metric_set_type == 1: val_metric_count += 1
            if metric_type > 2: unknown_type_count += 1

            # add new lines if model added new metric
            if graph_metric not in self.current_plot_lines:
                new_plot, = self.current_plot_ax.plot([], [], lw=metric_line_width, zorder=(graph_index+1)*10, label=graph_metric, marker=metric_marker, markersize=self.marker_size, markevery=(1, self.horizontal_grid_steps), color=metric_color, linestyle=metric_line_style)
                self.current_plot_lines[graph_metric] = new_plot

            used_metric_data = self.get_metric_averages(lightning_model, graph_metric)

            # update data for this metric
            self.current_plot_lines[graph_metric].set_data(range(self.starting_epoch - 1, self.starting_epoch + len(used_metric_data) - 1), used_metric_data)
            if self.use_style: self._decorate_line(self.current_plot_lines[graph_metric])

            # annotate (apparently this has to be done manually for every point)
            for index, data_point in enumerate(used_metric_data):
                # annotate on grid lines, and always annotate first and last data point
                if (index - 1) % self.horizontal_grid_steps != 0 and (self.starting_epoch + index - 1) != self.max_epochs and index != 0: continue

                used_data_point = None
                used_desc = None
                text_offset = None
                
                # Assume this is an intact data point
                if isinstance(data_point, (int, float, complex)) and not np.isnan(data_point): 
                    used_data_point = data_point
                    used_desc = str(round(data_point, 3))
                    text_offset = (-6,5)

                # Assuming None is set intentionally to leave this one out
                elif data_point == None:
                    used_data_point = float('nan')
                    used_desc = ""
                    text_offset = (0,0)

                # Assume everything else is an error that should be labeled as such and replace them with interpolation
                else:
                    #used_data_point = ((self.max_epochs - index)*0.7) / self.max_epochs #used_metric_data[index -1] if len(used_metric_data) > index and isinstance(used_metric_data[index -1], (int, float, complex)) else 0.5
                    preceding_data_point = used_metric_data[index -1] if (index -1) > 0 else None
                    succeeding_data_point = used_metric_data[index +1] if len(used_metric_data) > (index + 1) else None
                    if preceding_data_point == None and succeeding_data_point != None: used_data_point = succeeding_data_point
                    elif succeeding_data_point == None and preceding_data_point != None: used_data_point = preceding_data_point
                    elif succeeding_data_point == None and preceding_data_point == None: used_data_point = 0.5
                    else: used_data_point = preceding_data_point + (succeeding_data_point - preceding_data_point) * 0.5
                    text_offset = (-8,-2)

                    used_desc = "Error"
                self.current_annotations.append(self.current_plot_ax.annotate(used_desc, (index + self.starting_epoch - 1, used_data_point), fontsize=7, xytext=text_offset, textcoords="offset points", zorder=100))
                

        if last_metric_count != len(self.current_plot_lines): 
            #self.current_plot_figure.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, fancybox=True, shadow=True, frameon=True, borderpad=0.6, labelspacing=0, handleheight=-0.3, framealpha=0.5, facecolor=(0.094, 0.098, 0.101, 0.6), columnspacing=1.9, handletextpad=0.6, edgecolor=(0.094, 0.098, 0.101, 1))
            self.current_plot_figure.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, fancybox=True, shadow=True, frameon=True, borderpad=0.6, labelspacing=0, handleheight=-0.3, framealpha=1, facecolor=(0.094, 0.098, 0.101, 1), columnspacing=1.9, handletextpad=0.6, edgecolor=(0.094, 0.098, 0.101, 1))

        if self.live_show_graph: plt.pause(0.1) 

    def _set_plot_style(self, plot_ax=None, plot_fig=None, bright_text=False):
        if plot_ax == None: plot_ax = self.current_plot_ax
        if plot_fig == None: plot_fig = self.current_plot_figure

        plt.style.use("seaborn-dark")
        #bg_color = 0.094, 0.098, 0.101, 0.6
        bg_color = 0.094, 0.098, 0.101, 1.0
        tickAndLabelColor = 0.094, 0.098, 0.101, 1.0
        spinecolor = (1,1,1,0.75)
        if bright_text == True:
            tickAndLabelColor = "white"
            spinecolor="white"
        
        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = (0.094, 0.098, 0.101, 0) #'#48494B' 
        for param in ['text.color', 'axes.labelcolor']:
            plt.rcParams[param] = '0.99'
        for param in ['xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.66'

        if plot_ax != None and plot_fig != None:
            plot_ax.grid(color='#666666') 
            plot_ax.set_facecolor(bg_color)
            plot_fig.set_facecolor(bg_color)

            plot_ax.xaxis.label.set_color(tickAndLabelColor)
            plot_ax.yaxis.label.set_color(tickAndLabelColor)
            if hasattr(plot_ax, "zaxis"): plot_ax.zaxis.label.set_color(tickAndLabelColor)

            plot_ax.tick_params(axis='x', colors=tickAndLabelColor)
            plot_ax.tick_params(axis='y', colors=tickAndLabelColor)
            if hasattr(plot_ax, "zaxis"): plot_ax.tick_params(axis='z', colors=tickAndLabelColor)

            if hasattr(plot_ax, "zaxis"): 
                plot_ax.xaxis.pane.fill = False
                plot_ax.yaxis.pane.fill = False
                plot_ax.zaxis.pane.fill = False

            if hasattr(plot_ax, "zaxis"):
                plot_ax.zaxis.pane.set_edgecolor('w')
                plot_ax.xaxis.pane.set_edgecolor('w')
                plot_ax.yaxis.pane.set_edgecolor('w')


            plot_ax.spines['left'].set_color(spinecolor)
            plot_ax.spines['bottom'].set_color(spinecolor)



    def _decorate_line(self, drawn_line, drawing_ax=None, glow_size=None, glow_color=None, use_color_change=True):
        if drawing_ax == None: drawing_ax = self.current_plot_ax

        n_lines = 8
        alpha_value = 0.025
        original_color = matplotlib.colors.to_rgb(drawn_line.get_color() if glow_color == None else glow_color)
        original_linewidth = drawn_line.get_lw()
        diff_linewidth = original_linewidth/1.5 if glow_size == None else glow_size
        color_change = (((1 - (sum(original_color)/3))/n_lines) / 2) if use_color_change == True else 1

        for n in range(1, n_lines+1):
            i_color = tuple([min(1, color_component + (color_change*n)) for color_component in original_color])
            i_alpha = alpha_value
            if n == 1: 
                i_color = "#000000"
                i_alpha = 0.2

            new_deco_plot, = drawing_ax.plot(
                drawn_line.get_xdata(True),
                drawn_line.get_ydata(True),
                marker=drawn_line.get_marker(),
                linewidth=original_linewidth+(diff_linewidth*n),
                alpha=i_alpha,
                zorder=drawn_line.zorder - n,
                markevery=(1, self.horizontal_grid_steps),
                markersize=self.marker_size,
                #legend=False,
                color=i_color
            )
            if drawing_ax == self.current_plot_ax: self.current_decorations.append(new_deco_plot)

    def _remove_decorations(self):     
        removed_decos = []
        for ind, deco in enumerate(self.current_decorations):
            self.current_plot_ax.lines.remove(deco)
            removed_decos.append(deco)
        
        for removed_deco in removed_decos:
            self.current_decorations.remove(removed_deco)

    def _remove_annotations(self):
        removed_annos = []
        for ind, anno in enumerate(self.current_annotations):
            anno.remove()
            removed_annos.append(anno)
        
        for removed_anno in removed_annos:
            self.current_annotations.remove(removed_anno)
    
    def get_experiment_string(self, file_friendly=False, overwrite_version=None):
        version = self.current_experiment_version 
        if overwrite_version != None: version = overwrite_version
        if file_friendly == False: return (f"{self.model_name}  |  {self.data_procurer_name}  |  Version {version} - {datetime.date.today().strftime('%d.%m.%Y')}").replace("_", " ").title()
        if file_friendly == True: return (f"{self.model_name}-{self.data_procurer_name}-Version_{version}_{datetime.date.today().strftime('%d.%m.%Y')}").title()


    def set_experiment_info(self, log_dir, experiment_version, model_name, data_procurer_name, loading_locally, filename=None):
        self.logging_directory = log_dir
        self.current_experiment_version = experiment_version
        self.model_name = model_name
        self.data_procurer_name = data_procurer_name
        self.loading_locally = loading_locally
        if filename != None: self.text_name = filename + self.text_name

    def set_starting_epoch(self, starting_epoch):
        self.starting_epoch = starting_epoch

    
    def get_metric_averages(self, lightning_model, metric_name):
        if not lightning_model.visual_metrics or lightning_model.visual_metrics.get(metric_name, None) == None: return None

        metric_data = lightning_model.visual_metrics[metric_name]

        # start data at epoch 0 or epoch -1, depending on when data tracking started
        extra_data = 0
        if len(self.extra_sanity_data) > 0 and self.extra_sanity_data.get(metric_name, None) != None: 
            extra_data = self.extra_sanity_data[metric_name]

        # calculate how much data there is per epoch
        #data_per_epoch = len(metric_data)/(lightning_model.current_epoch + 1 + (1 if metric_start_point == -1 else 0))
        data_per_epoch = (len(metric_data) - extra_data)/(lightning_model.current_epoch + 1)
        if data_per_epoch != int(data_per_epoch): 
            self.critical("Model added varying amounts of visual metric data per epoch. This may lead to inconsistent graphs as all results are averaged per epoch, assuming equal distribution.")
        data_per_epoch = int(data_per_epoch)
        
        used_metric_data = metric_data[extra_data:]

        # if there is more data than epochs, assume the data is given per training step and must be averaged per epoch
        if len(used_metric_data) > (lightning_model.current_epoch + 1):
            #used_metric_data_format1 = [np.mean(i_graph_metric_data[i:i + batches_per_epoch]) for i in range(0, len(i_graph_metric_data), batches_per_epoch)]
            used_metric_data = [np.nanmean(metric_data[i:i + data_per_epoch]) for i in range(extra_data, len(metric_data), data_per_epoch)]

        # prepend metric data that wasn't given per epoch
        if extra_data != 0:
            used_metric_data = [np.nanmean(metric_data[0:extra_data])] + used_metric_data
        else:
            used_metric_data = [None] + used_metric_data

        return used_metric_data

    """
    def _update_metric_sizes(self, lightning_model):
        for graph_metric in lightning_model.visual_metrics:
            i_graph_metric_data = lightning_model.visual_metrics[graph_metric]

            # add tracking for new metrics
            if self.metric_size_tracker.get(graph_metric, None) == None:
                self.metric_size_tracker[graph_metric] = []
            
            # calculated amount of added data for this epoch by comparing current amount of data to previous ones
            n_added_data = len(i_graph_metric_data) - sum(self.metric_size_tracker[graph_metric])
    """
    def _get_full_overview_path(self):
        return os.path.join(self._get_experiments_parent_directory(self.logging_directory), self.version_overview_name)
        

    def _get_experiments_parent_directory(self, experiment_directory):
        #directories = os.path.split(experiment_directory)
        #result = directories[0:-2]
        #result = os.path.join(*result) #if len(result) > 1 else result[0]

        result = pkg_resources.resource_filename("tud_presence_prediction",self.base_results_path)

        return result

    def _get_current_overview_line(self, lightning_model):
        overview_line = ""
        overview_line += self.model_name + ","
        overview_line += self.data_procurer_name + ","
        overview_line += str(self.loading_locally) + ","
        overview_line += str(self.current_experiment_version) + ","
        overview_line += str(lightning_model.current_epoch)

        if lightning_model and lightning_model.visual_metrics: 
            for overview_metric in self.default_metrics:
                last_epoch_data = None
                data_per_epoch = self.get_metric_averages(lightning_model, overview_metric)
                if data_per_epoch != None and len(data_per_epoch) > 0:
                    last_epoch_data = data_per_epoch[-1]
                    # get last average for data
                    #data_per_epoch = int(len(lightning_model.visual_metrics[overview_metric])/(lightning_model.current_epoch + 1 + (1 if self.extra_sanity_data.get(overview_metric, None) != None == -1 else 0)))
                    #last_epoch_data = lightning_model.visual_metrics[overview_metric][-data_per_epoch-1:-1]
                    #last_epoch_data = "None" if len(last_epoch_data) < 1 else str(np.mean(last_epoch_data))
                overview_line += "," + str(last_epoch_data)

        return overview_line

    def _add_training_session_to_version_overview(self, lightning_model):
        if self.mode < logger_mode.TEXT: return
        
        full_content = ""
        overview_filename = self._get_full_overview_path()

        # create overview if it doesnt exist
        if not os.path.exists(overview_filename):
            full_content += "sep=," + "\n"
            full_content += "Model,Data Procurer,Local Data,Version,Epoch,"
            full_content += ",".join([metric_header.capitalize() for metric_header in self.default_metrics])
            
        full_content += "\n"
        full_content += self._get_current_overview_line(lightning_model)
        
        os.makedirs(os.path.dirname(overview_filename), exist_ok=True)
        with open(overview_filename, 'a+', encoding="UTF-8") as overview_file:
            overview_file.write(full_content)


        
    def _update_version_overview(self, lightning_model):
        if self.mode < logger_mode.TEXT: return

        overview_filename = self._get_full_overview_path()

        current_line = self._get_current_overview_line(lightning_model)

        with open(overview_filename, "r+", encoding = "utf-8") as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell() - 1

            while pos > 0 and file.read(1) != "\n":
                pos -= 1
                file.seek(pos, os.SEEK_SET)

            if pos > 0:
                file.seek(pos - 1, os.SEEK_SET)
                file.truncate()

            file.write("\n" + current_line)


    def _save_sanity_data(self, lightning_model):
        if not lightning_model or not lightning_model.visual_metrics: return
        if self.mode < logger_mode.FULL: return

        for graph_metric in lightning_model.visual_metrics:
            i_graph_metric_data = lightning_model.visual_metrics[graph_metric]

            if len(i_graph_metric_data) != 0: self.extra_sanity_data[graph_metric] = len(i_graph_metric_data)

    # --- internal logging methods ---
    def _console_log(self, text, new_line=False):
        if self.mode >= logger_mode.CONSOLE: 
            if new_line: print()
            print(text)

    def _file_log(self, text, new_line=False):
        if self.logging_directory == None: return
        if self.mode < logger_mode.TEXT: return

        text_to_write = ""
        if new_line: text_to_write = "\n"
        text_to_write += text + "\n"

        self.full_log += text_to_write

        os.makedirs(self.logging_directory, exist_ok=True) # create directory if it hasn't been created yet (by lightning)
        with open(self.logging_directory + os.path.sep + self.text_name, 'w+') as log_file:
            log_file.write(self.full_log)

    # --- callbacks from lightning ---
    def on_train_start(self, lightning_trainer, lightning_model):
        self._add_training_session_to_version_overview(lightning_model)
        print("")

    def on_sanity_check_end(self, lightning_trainer, lightning_model):
         if self.live_show_graph or self.live_save_graph:
            self._save_sanity_data(lightning_model)

    def on_train_epoch_end(self, lightning_trainer, lightning_model):
        # update graph if its active
        if self.live_show_graph or self.live_save_graph:
            self._update_training_graph(lightning_model)
            if self.current_plot_figure != None: self.save_training_graph()

        # update version overview
        self._update_version_overview(lightning_model)

        # Automatic console outputs
        cur_epoch = lightning_model.current_epoch
        last_train_loss = self.get_metric_averages(lightning_model, "Training loss")
        last_train_loss = round(last_train_loss[-1], 15) if last_train_loss != None and len(last_train_loss) > 1 else None
        last_val_loss = self.get_metric_averages(lightning_model, "Validation loss")
        last_val_loss = round(last_val_loss[-1], 15) if last_val_loss != None and len(last_val_loss) > 1 else None
        self.info(f" Epoch {cur_epoch} ended --- Training loss: {last_train_loss} --- Validation loss: {last_val_loss} ")

        if self.use_time_profiler:
            self.info(f"Timings:")
            time_results = TimeProfiler.get_averages()
            for section in time_results:
                self.info(f"{(section['section_name'] + ':'):<40} {section['duration'] * 1000} ms average, {section['measurement_count']} records.")
            if TimeProfiler.max_measurements == -1:
                TimeProfiler.clear() # clear all measurements between epochs unless profiler deletes automatically


    def on_train_batch_end(self, lightning_trainer, lightning_model, outputs, batch, batch_idx):
        #plt.pause(0.01)
        return

    


class logger_mode(Enum):
    FULL = 8
    TEXT = 4
    CONSOLE = 2
    NONE = 1

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

