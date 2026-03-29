import csv
import random
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.tree import _tree
import numpy as np

plantDict = {}

with open('resources/plant_conditions.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    next(reader)
    
    i = 0

    for row in reader:
        plantDict[i] = [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
        i += 1


X = []
y = []

for plantIndex in plantDict:
    plant = plantDict[plantIndex]
    
    # 50 samples INSIDE ideal range — should score 85-90
    for i in range(50):
        plant_light_min = 5500 + (plant[5] / 30000) * (50000 - 5500)
        plant_light_max = 5500 + (plant[6] / 30000) * (50000 - 5500)
        temp = random.uniform(plant[1], plant[2])
        humidity = random.uniform(plant[3], plant[4])
        light_val = random.uniform(plant_light_min, plant_light_max)
        score = random.uniform(85, 90)
        X.append([plantIndex, temp, humidity, light_val])
        y.append(score)
    
    # 50 samples SLIGHTLY outside — should score 40-70
    for i in range(50):
        plant_light_min = 5500 + (plant[5] / 30000) * (50000 - 5500)
        plant_light_max = 5500 + (plant[6] / 30000) * (50000 - 5500)
        temp = random.uniform(plant[1] - 15, plant[2] + 15)
        humidity = random.uniform(max(0, plant[3] - 20), min(100, plant[4] + 20))
        light_val = random.uniform(max(5500, plant_light_min - 10000), min(50000, plant_light_max + 10000))
        
        t_off = max(0, plant[1] - temp, temp - plant[2])
        h_off = max(0, plant[3] - humidity, humidity - plant[4])
        l_off = max(0, plant_light_min - light_val, light_val - plant_light_max)
        
        t_pct = t_off / max(1, plant[2] - plant[1]) 
        h_pct = h_off / max(1, plant[4] - plant[3])
        l_pct = l_off / max(1, plant_light_max - plant_light_min)
        
        penalty = (t_pct + h_pct + l_pct) / 3
        score = max(30, 85 - (penalty * 55))
        X.append([plantIndex, temp, humidity, light_val])
        y.append(score)
    
    # 50 samples FAR outside — should score 0-30
    for i in range(75):
        plant_light_min = 5500 + (plant[5] / 30000) * (50000 - 5500)
        plant_light_max = 5500 + (plant[6] / 30000) * (50000 - 5500)
        temp = random.uniform(30, 110)
        humidity = random.uniform(0, 100)
        light_val = random.uniform(5500, 50000)
        
        t_off = max(0, plant[1] - temp, temp - plant[2])
        h_off = max(0, plant[3] - humidity, humidity - plant[4])
        l_off = max(0, plant_light_min - light_val, light_val - plant_light_max)
        
        t_pct = t_off / max(1, plant[2] - plant[1])
        h_pct = h_off / max(1, plant[4] - plant[3])
        l_pct = l_off / max(1, plant_light_max - plant_light_min)
        
        penalty = (t_pct + h_pct + l_pct) / 3
        score = max(0, 30 - (penalty * 35))
        X.append([plantIndex, temp, humidity, light_val])
        y.append(score)

clf = DecisionTreeRegressor(max_depth=10)
clf.fit(X, y)

def tree_to_python(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    lines = []
    lines.append("def predict(plant, temp, humidity, light):")

    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 2)
            lines.append(f"{indent}if {name} <= {threshold}:")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}else:")
            recurse(tree_.children_right[node], depth + 1)
        else:
            value = round(tree_.value[node][0][0], 2)
            lines.append(f"{indent}return {value}")

    recurse(0, 0)
    return "\n".join(lines)

python_code = tree_to_python(clf, ['plant', 'temp', 'humidity', 'light'])

with open('src/model.py', 'w') as f:
    f.write(python_code)

print("Model exported to model.py")