from tud_presence_prediction.presence_prediction import presence_prediction

action = "train"                    # action to perform (see below)
version = "3"                       # local trained version of the model; needed for prediction and evaluation


if action == "train":

    test_presence_prediction = presence_prediction(
        model_file="Transformer_V2_regressive",
        data_procurer_file = "dynamic_multiuser",
        batch_size=10,              # should be adapted to the accelerator
        sequence_size=[16,],        
        stride_size=1,              

        logging_mode="full",        # may be switched to "text" if image generation causes issues on servers
        extrapolation="none",       
        epochs=150                  # exemplary value
    )

    test_presence_prediction.train()

elif action == "predict with download":

    test_presence_prediction = presence_prediction(
        model_file="Transformer_V2_regressive",
        data_procurer_file = "dynamic_singleuser",  
        sequence_size=[16,],
        stride_size=1,

        extrapolation="now",        
        version=version,
        days=3,                 

        # users have to be defined in the discussed text file or environment variable and are then procured via HTTP
        user=4
    )

    test_presence_prediction.predict()

elif action == "predict with local data":

    test_presence_prediction = presence_prediction(
        model_file="Transformer_V2_regressive",
        data_procurer_file = "dynamic_singleuser",
        sequence_size=[16,],
        stride_size=8,

        extrapolation="now",        
        version=version,
        days=3,

        # data below is purely exemplary
        user_home_coordinates='{"coordinate": [12298, 19260]}',
        user_data='[{"timestamp":1683710100000,"coordinate":[12298,19260]},{"timestamp":1683711000000,"coordinate":[12298,19260]},{"timestamp":1683711900000,"coordinate":[12298,19260]},{"timestamp":1683712800000,"coordinate":[12298,19260]},{"timestamp":1683713700000,"coordinate":[12298,19260]},{"timestamp":1683714600000,"coordinate":[12291,19260]},{"timestamp":1683715500000,"coordinate":[12298,19264]},{"timestamp":1683716400000,"coordinate":[12298,19260]},{"timestamp":1683717300000,"coordinate":[12291,19260]},{"timestamp":1683718200000,"coordinate":[18562,16315]},{"timestamp":1683719100000,"coordinate":[18541,19444]},{"timestamp":1683720000000,"coordinate":[12291,19264]},{"timestamp":1683720900000,"coordinate":null},{"timestamp":1683721800000,"coordinate":[12291,19268]},{"timestamp":1683722700000,"coordinate":[12291,19260]},{"timestamp":1683723600000,"coordinate":[12291,19260]},{"timestamp":1683724500000,"coordinate":[12298,19260]},{"timestamp":1683725400000,"coordinate":[12291,19260]},{"timestamp":1683726300000,"coordinate":[12291,19260]},{"timestamp":1683727200000,"coordinate":[12298,19260]},{"timestamp":1683728100000,"coordinate":null},{"timestamp":1683729000000,"coordinate":[12291,19260]},{"timestamp":1683729900000,"coordinate":[24406,6838]},{"timestamp":1683730800000,"coordinate":null},{"timestamp":1683731700000,"coordinate":null},{"timestamp":1683732600000,"coordinate":[12291,19260]},{"timestamp":1683733500000,"coordinate":null},{"timestamp":1683734400000,"coordinate":[12291,19256]},{"timestamp":1683735300000,"coordinate":null},{"timestamp":1683736200000,"coordinate":[12291,19264]},{"timestamp":1683737100000,"coordinate":[12291,19260]},{"timestamp":1683738000000,"coordinate":[12291,19264]},{"timestamp":1683738900000,"coordinate":[12291,19260]},{"timestamp":1683739800000,"coordinate":[12298,19264]},{"timestamp":1683740700000,"coordinate":[12298,19264]},{"timestamp":1683741600000,"coordinate":[12298,19260]},{"timestamp":1683742500000,"coordinate":[12298,19260]},{"timestamp":1683743400000,"coordinate":[12298,19260]},{"timestamp":1683744300000,"coordinate":[12298,19260]},{"timestamp":1683745200000,"coordinate":[12298,19260]},{"timestamp":1683746100000,"coordinate":[12298,19260]},{"timestamp":1683747000000,"coordinate":[6237,22491]},{"timestamp":1683747900000,"coordinate":[12298,19260]},{"timestamp":1683748800000,"coordinate":[12298,19260]},{"timestamp":1683749700000,"coordinate":null},{"timestamp":1683750600000,"coordinate":null},{"timestamp":1683751500000,"coordinate":null},{"timestamp":1683792000000,"coordinate":[12298,19264]},{"timestamp":1683792900000,"coordinate":[12298,19260]},{"timestamp":1683793800000,"coordinate":[12298,19260]},{"timestamp":1683794700000,"coordinate":[12298,19260]},{"timestamp":1683795600000,"coordinate":[12298,19260]},{"timestamp":1683796500000,"coordinate":[12298,19260]},{"timestamp":1683797400000,"coordinate":[12291,19260]},{"timestamp":1683798300000,"coordinate":[12291,19260]},{"timestamp":1683799200000,"coordinate":[12291,19260]},{"timestamp":1683800100000,"coordinate":[12291,19260]},{"timestamp":1683801000000,"coordinate":[12291,19264]},{"timestamp":1683801900000,"coordinate":[12291,19264]},{"timestamp":1683802800000,"coordinate":[12291,19264]},{"timestamp":1683803700000,"coordinate":[12291,19264]},{"timestamp":1683804600000,"coordinate":[24616,28835]},{"timestamp":1683805500000,"coordinate":[31048,22661]},{"timestamp":1683806400000,"coordinate":null},{"timestamp":1683807300000,"coordinate":[12298,19260]},{"timestamp":1683808200000,"coordinate":null},{"timestamp":1683809100000,"coordinate":[12298,19260]},{"timestamp":1683810000000,"coordinate":[12298,19260]},{"timestamp":1683810900000,"coordinate":[12298,19260]},{"timestamp":1683811800000,"coordinate":[12298,19260]},{"timestamp":1683812700000,"coordinate":[12298,19260]},{"timestamp":1683813600000,"coordinate":[12298,19260]},{"timestamp":1683814500000,"coordinate":[12298,19260]},{"timestamp":1683815400000,"coordinate":[12298,19260]},{"timestamp":1683816300000,"coordinate":null},{"timestamp":1683817200000,"coordinate":[12291,19260]},{"timestamp":1683818100000,"coordinate":[12298,19260]},{"timestamp":1683819000000,"coordinate":[12298,19260]},{"timestamp":1683819900000,"coordinate":[6055,16217]},{"timestamp":1683820800000,"coordinate":[-195,22483]},{"timestamp":1683821700000,"coordinate":[12298,19260]},{"timestamp":1683822600000,"coordinate":[12298,19260]},{"timestamp":1683823500000,"coordinate":[12298,19260]},{"timestamp":1683824400000,"coordinate":[12298,19260]},{"timestamp":1683825300000,"coordinate":[12298,19260]},{"timestamp":1683826200000,"coordinate":[12298,19260]},{"timestamp":1683827100000,"coordinate":null},{"timestamp":1683828000000,"coordinate":[12298,19260]},{"timestamp":1683828900000,"coordinate":[12298,19260]},{"timestamp":1683829800000,"coordinate":[12298,19260]},{"timestamp":1683830700000,"coordinate":[12298,19264]},{"timestamp":1683831600000,"coordinate":null},{"timestamp":1683832500000,"coordinate":null},{"timestamp":1683833400000,"coordinate":[17981,31067]},{"timestamp":1683834300000,"coordinate":[17981,6067]},{"timestamp":1683835200000,"coordinate":[17981,6067]},{"timestamp":1683836100000,"coordinate":null},{"timestamp":1683837000000,"coordinate":null},{"timestamp":1683837900000,"coordinate":null},{"timestamp":1683878400000,"coordinate":[12298,19260]},{"timestamp":1683879300000,"coordinate":[12298,19260]},{"timestamp":1683880200000,"coordinate":[12291,19264]},{"timestamp":1683881100000,"coordinate":[12298,19260]},{"timestamp":1683882000000,"coordinate":[18548,25698]},{"timestamp":1683882900000,"coordinate":[12298,19260]},{"timestamp":1683883800000,"coordinate":[12298,19260]},{"timestamp":1683884700000,"coordinate":[12291,19264]},{"timestamp":1683885600000,"coordinate":null},{"timestamp":1683886500000,"coordinate":[12298,19260]},{"timestamp":1683887400000,"coordinate":[18163,25510]},{"timestamp":1683888300000,"coordinate":[18170,510]},{"timestamp":1683889200000,"coordinate":null},{"timestamp":1683890100000,"coordinate":[24595,-11888]},{"timestamp":1683891000000,"coordinate":[12298,19260]},{"timestamp":1683891900000,"coordinate":[12298,19264]},{"timestamp":1683892800000,"coordinate":[12298,19264]},{"timestamp":1683893700000,"coordinate":[12298,19260]},{"timestamp":1683894600000,"coordinate":null},{"timestamp":1683895500000,"coordinate":null},{"timestamp":1683896400000,"coordinate":[24798,19630]},{"timestamp":1683897300000,"coordinate":[31062,22665]},{"timestamp":1683898200000,"coordinate":null},{"timestamp":1683899100000,"coordinate":[12291,19260]},{"timestamp":1683900000000,"coordinate":[18548,25698]},{"timestamp":1683900900000,"coordinate":null},{"timestamp":1683901800000,"coordinate":[12291,19264]},{"timestamp":1683902700000,"coordinate":null},{"timestamp":1683903600000,"coordinate":[12291,19264]}]'
    )


    test_presence_prediction.predict()

elif action == "evaluate":

    test_presence_prediction = presence_prediction(
        model_file="Transformer_V2_regressive",
        data_procurer_file = "dynamic_multiuser",
        sequence_size=[16,],
        stride_size=1,

        extrapolation="none", 
        version=version
    )

    # Exemplary settings: window_size_step=sequence_size, window_offset_step=1 (maximum variations), prediction_length_step=1, prediction_length_max=3.
    test_presence_prediction.evaluate(window_size_step=16, window_offset_step=1, prediction_length_step=1, prediction_length_max=1)
