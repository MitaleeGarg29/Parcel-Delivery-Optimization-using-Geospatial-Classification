import time
class TimeProfiler():
    """
    Provides rudimentary timing profiling functionality.

    Execution times of any code segments may be accumelated and averaged by enclosing them as follows:
    TimeProfiler.begin('custom section name')
    # your code
    TimeProfiler.end('custom section name')

    By default, the results are averaged and logged every epoch if Time profiling is enabled in the presence prediction. 
    If desired, results can optionally be generated at any time using get_average() or get_averages().

    This class serves no productive purpose and is only intended to be used as a developer tool.
    """

    active = True

    max_measurements = -1

    beginning_measurements = dict()     # dict[list[float]]
    end_measurements = dict()           # dict[list[float]]

    @staticmethod
    def begin(section_name):
        if TimeProfiler.active == False: return

        if TimeProfiler.beginning_measurements.get(section_name, None) == None:
            TimeProfiler.beginning_measurements[section_name] = list()
        elif TimeProfiler.max_measurements != -1 and len(TimeProfiler.beginning_measurements[section_name]) >= TimeProfiler.max_measurements:
            TimeProfiler.beginning_measurements[section_name].pop(0)

        TimeProfiler.beginning_measurements[section_name].append(time.time())

    @staticmethod
    def end(section_name):
        if TimeProfiler.active == False: return

        if TimeProfiler.end_measurements.get(section_name, None) == None:
            TimeProfiler.end_measurements[section_name] = list()
        elif TimeProfiler.max_measurements != -1 and len(TimeProfiler.end_measurements[section_name]) >= TimeProfiler.max_measurements:
            TimeProfiler.end_measurements[section_name].pop(0)

        
        TimeProfiler.end_measurements[section_name].append(time.time())

    @staticmethod
    def get_averages():
        results = []
        for section in TimeProfiler.end_measurements:
            results.append(TimeProfiler.get_average(section))
        
        return results

    @staticmethod
    def get_average(section_name):
        result = dict()
        result["section_name"] = section_name
        result["duration"] = 0
        result["measurement_count"] = len(TimeProfiler.end_measurements[section_name])
        for index, end in enumerate(TimeProfiler.end_measurements[section_name]):
            result["duration"] += (end - TimeProfiler.beginning_measurements[section_name][index]) / result["measurement_count"]
        return result
    
    @staticmethod
    def clear():
        TimeProfiler.beginning_measurements = dict()     # dict[list[float]]
        TimeProfiler.end_measurements = dict()           # dict[list[float]]



        
    