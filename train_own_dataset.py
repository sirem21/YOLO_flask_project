import json
from ultralytics import YOLO
import cv2 as cv
from collections import Counter
# train YOLO model with own dataset
def train_model():
    print("YOLOv8 model is being loaded and training is starting...")
    # Configure from YAML and transfer weights
    model1 = YOLO("yolov8n.yaml").load("yolov8n.pt")
    #process will go through the entire training dataset 50 time
    results = model1.train(data="D:\\PythonProject YOLO\\local_env\\config.yaml", epochs=30, imgsz=640)
    print("Model training completed.")
#train_model()

# object detection
def predicted_image(model,image_path):
    img = cv.imread(image_path)
    # find number of found objects
    results = model.predict(image_path, conf=0.25, save=True)
    boxes = results[0].boxes
    print("Number of boxes:", len(boxes))

    #for each labeled objects print
    for b in boxes:
        print(" Predicted labeled class:", int(b.cls[0]),
              " Confidence of truth:", float(b.conf[0]))
    return results


def predict_nutrition_values(results,model,nutritionValue_dict_per_100gr):
    boxes = results[0].boxes
    detected_items = []
    total_calories = 0
    vitamins_facts = {
        'apple':
            "Apples are a good source of dietary fiber and vitamin C. They also contain antioxidants like quercetin and flavonoids, which support heart health and may help reduce inflammation. Low in calories and high in water, apples make a great snack for digestive and immune support.",
        'orange':
            "Oranges are rich in vitamin C, essential for immune function, skin health, and iron absorption. They also contain fiber, potassium, and antioxidants like flavonoids, which can help lower blood pressure and improve heart health.",
        'pineapple':
            "Pineapples are loaded with vitamin C and manganese. They also contain bromelain, an enzyme that helps with digestion and may reduce inflammation. Their sweet, tropical flavor comes with immune-boosting and anti-inflammatory benefits.",
        'strawberry':
            "Strawberries are an excellent source of vitamin C, manganese, folate, and antioxidants such as anthocyanins. They support skin health, reduce oxidative stress, and may help regulate blood sugar levels due to their low glycemic templates.",
        'watermelon':
            "Watermelon is an excellent natural source of vitamin C, providing about 10–14 mg per 100 g (around 12.5 mg per cup) — that’s roughly 14 % of the daily value. This antioxidant supports immune health, skin repair, and helps with absorption of iron ."
    }
    for b in boxes:
        c = b.cls
        class_name = model.names[int(c)]
        detected_items.append(class_name)

    # count number of each type of fruits
    counter = Counter(detected_items)
    fruit_list = []
    counter_list = []
    for fruit, count in counter.items():
        counter_list.append(count)
        fruit_list.append(fruit)
        print(f"Detected Items: {fruit}  Number of each type: {count}")
    facts_list = []
    cal_list = []
    prot_list = []
    carbo_list = []
    fats_list = []
    result_json = []
    # detect unique classes list
    unic_names = list(set(detected_items))
    # for each type of fruit write its own nutrition values
    for n in unic_names:
        facts_list.append(vitamins_facts.get(n, 0))
        cal_list.append(nutritionValue_dict_per_100gr.get(n, {}).get('kcal', 0))
        prot_list.append(nutritionValue_dict_per_100gr.get(n, {}).get('proteins', 0))
        carbo_list.append(nutritionValue_dict_per_100gr.get(n, {}).get('carbohydrates', 0))
        fats_list.append(nutritionValue_dict_per_100gr.get(n, {}).get('fats', 0))
    for i in range(len(unic_names)):
        print(f"Nutrition Values per 100gr of a {unic_names[i]} \n"
          f"Calories : {cal_list[i]} \n"
          f"Proteins : {prot_list[i]} \n"
          f"Carbohydrates: {carbo_list[i]} \n"
          f"Fats: {fats_list[i]} \n"
          f"Vitamin Info: {facts_list[i]}\n")

        fruitInfo_json = {
            "name" : unic_names[i],
            "count" : counter_list[i],
            "calories": cal_list[i],
            "proteins": prot_list[i],
            "carbohydrates": carbo_list[i],
            "fats": fats_list[i],
            "vitamins": facts_list[i]
        }
        result_json.append(fruitInfo_json)
    return result_json

def main(image_path):
    #train_model()
    nutritionValue_dict_per_100gr = {
        'apple': {'kcal': 52, 'proteins':0.3, 'carbohydrates':13.8, 'fats': 0.2 },
        'orange': {'kcal': 47, 'proteins':0.9, 'carbohydrates':11.8, 'fats': 0.1 },
        'pineapple': {'kcal': 50, 'proteins':0.5, 'carbohydrates':13.1, 'fats': 0.1 },
        'strawberry': {'kcal': 32, 'proteins':0.8, 'carbohydrates':7.7, 'fats': 0.3 },
        'watermelon': {'kcal': 30, 'proteins':0.6, 'carbohydrates':7.6, 'fats': 0.1 }
    }
    model = YOLO("runs/detect/train2/weights/best.pt")
    results = predicted_image(model,image_path)
    nutrition_values = predict_nutrition_values(results,model,nutritionValue_dict_per_100gr)

    return nutrition_values
#if __name__ == '__main__':
#    main('')
