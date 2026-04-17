from presence_prediction import presence_prediction
import requests

if __name__ == "__main__":
    usr_id = "7cee070fdc0cf380359e432f06d4d1cc1359267334f7693c034a430e4bde3499"  #'4147127f-1c47-4de8-8e43-58f19d41b418'
    geo_hash_home = "duzz2"
    # Get data from database
    json_data = []
    urls = [
        "https://rl.green-convenience.com/api/v1/positions/convert/"
        + usr_id
        + "/"
        + geo_hash_home,
        "https://rl.green-convenience.com/api/v1/positions/preprocess/"
        + usr_id
        + "/"
        + geo_hash_home,
    ]
    home_address_response = requests.get(urls[0])
    data_response = requests.get(urls[1])
    home_address_text = ""
    data_text = ""
    if home_address_response.status_code == 200:
        home_address_text = home_address_response.text
    else:
        print(
            f"Failed to retrieve data from the URL. Status code: {home_address_response.status_code}"
        )
    if data_response.status_code == 200:
        data_text = data_response.text
    else:
        print(
            f"Failed to retrieve data from the URL. Status code: {home_address_response.status_code}"
        )

    test_presence_prediction = presence_prediction(
        model_file="LSTM_V1_longmem_batch",
        data_procurer_file="dynamic_singleuser",
        data_time_shift="2:0",
        sequence_size=[
            16,
        ],
        stride_size=8,
        extrapolation="now",
        logging_mode="none",
        version=1,
        days=2,
        user_home_coordinates=home_address_text,
        user_data=data_text,
        production=True,
        model_path_production="./training_results/LSTM_V1_longmem_batch/lightning_logs/version_1/checkpoints/training-epoch=0000600-best_loss_ckpt-train_loss=0.27.ckpt",
    )

    result = test_presence_prediction.predict()
    print(result)
