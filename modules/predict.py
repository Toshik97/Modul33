# <YOUR_IMPORTS>
import pandas as pd
import json
import os
import glob  # Добавлен импорт glob
from sklearn.ensemble import RandomForestClassifier  # Убедитесь, что это импортировано
import joblib
import dill
from datetime import datetime
import logging



def load_model(models_path): #вытаскиваем модель из списка ( там может быть несколько, я так случайно сделала)
    model_files = glob.glob(os.path.join(models_path, 'cars_pipe_*.pkl'))
    best_model_file = max(model_files, key=os.path.getmtime)
    with open(best_model_file, 'rb') as file:
        model = dill.load(file)
    return model


def load_test_data(test_data_path):
    test_files = glob.glob(os.path.join(test_data_path, '*.json'))
    return test_files

def predict_and_process(model, test_files):
    all_predictions = []
    for file in test_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:  # Явное указание кодировки
                data = json.load(f)

            file_name = os.path.basename(file)  # Получаем имя файла (например, "7316509996.json")
            car_id = file_name.replace('.json', '')  # Извлекаем ID (например, "7316509996")

            df = pd.DataFrame([data])



            predictions = model.predict(df)


            predictions_df = pd.DataFrame({'car_id': car_id, 'pred': predictions})
            all_predictions.append(predictions_df)

        except FileNotFoundError:  # Обработка ситуации, когда файл не найден
            print(f"Error: File not found: {file}")
        except json.JSONDecodeError:  # Обработка ошибок декодирования JSON
            print(f"Error: Invalid JSON in file: {file}")
        except Exception as e:  # Обработка любых других исключений
            print(f"Error processing file {file}: {e}")

    return pd.concat(all_predictions, ignore_index=True)


def save_predictions(final_predictions_df, predictions_path):
    """Сохраняет предсказания в CSV-файл."""
    os.makedirs(predictions_path, exist_ok=True)
    output_file = os.path.join(predictions_path, f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    try:
        final_predictions_df.to_csv(output_file, index=False, columns=['car_id', 'pred'])
        logging.info(f"Предсказания сохранены в {output_file}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении предсказаний: {e}")




def predict():
    # <YOUR_CODE>
    # загружает обученную модель

    models_path = 'data/models'  # Путь к папке с моделями
    test_data_path = 'data/test'
    predictions_path = 'data/predictions'

    model = load_model(models_path)
    test_files = load_test_data(test_data_path)
    final_predictions_df = predict_and_process(model, test_files)

    save_predictions(final_predictions_df, predictions_path)

if __name__ == '__main__':
    predict()






