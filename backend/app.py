from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from math import pi
import os
import xgboost as xgb

app = Flask(__name__, static_folder='static')

# 加载已保存的模型
model = joblib.load('backend/xgboost_model.pkl')

@app.route('/')
def index():
    # 返回根目录中的 index.html
    return send_from_directory('../', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从表单获取用户输入
        input_data = {
            'Sex': request.form.get('Sex'),
            'T_Stage': request.form.get('T_Stage'),
            'N_Stage': request.form.get('N_Stage'),
            'Age_Group': request.form.get('Age_Group'),
            'Primary_Site_Category': request.form.get('Primary_Site_Category'),
            'Grade_Group': request.form.get('Grade_Group'),
            'Tumor_histology': request.form.get('Tumor_histology'),
            'Tumor_size': request.form.get('Tumor_size'),
            'Number_of_nodes_examined': request.form.get('Number_of_nodes_examined'),
            'Surgery_Combined': request.form.get('Surgery_Combined'),
            'Primary_tumor': request.form.get('Primary_tumor')
        }

        # 检查是否有未选择的字段
        for key, value in input_data.items():
            if value is None or value == "":
                return jsonify({'error': f"错误: {key} 字段未选择。"})

        # 将输入数据转化为模型输入格式
        input_data_df = pd.DataFrame([list(input_data.values())], columns=input_data.keys())

        # 确保列顺序与模型训练时保持一致
        desired_order = ['Sex', 'T_Stage', 'N_Stage', 'Age_Group',
                         'Primary_Site_Category', 'Grade_Group',
                         'Tumor_histology', 'Tumor_size',
                         'Number_of_nodes_examined', 'Surgery_Combined',
                         'Primary_tumor']

        input_data_df = input_data_df[desired_order]  # 按训练数据的顺序排列列

        # 将输入数据转换为 NumPy 数组
        input_data_array = input_data_df.values

        # 将 NumPy 数组转换为 XGBoost 的 DMatrix 格式
        dmatrix_data = xgb.DMatrix(input_data_array)

        # 使用模型进行预测
        prediction = model.predict(dmatrix_data)

        # 根据预测值生成概率结果
        metastasis_proba = float(prediction[0]) * 100

        # 返回预测结果
        response = {
            "metastasis_proba": metastasis_proba,
            "prediction": int(prediction[0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"预测过程中发生错误: {str(e)}"})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
