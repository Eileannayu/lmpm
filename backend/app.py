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
            'Grade_Group': request.form.get('Grade_Group'),
            'Primary_Site_Category': request.form.get('Primary_Site_Category'),
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
        input_data_df = input_data_df.astype('category')

        # 确保输入数据的列顺序
        desired_order = ['Sex', 'T_Stage', 'N_Stage', 'Age_Group',
                        'Primary_Site_Category', 'Grade_Group',
                        'Tumor_histology', 'Tumor_size',
                        'Number_of_nodes_examined', 'Surgery_Combined',
                        'Primary_tumor']

        try:
            input_data_df = input_data_df[desired_order]
        except KeyError as e:
            return jsonify({'error': f"缺少必要的输入字段: {str(e)}"})

        input_data_array = input_data_df.values

        # 将 NumPy 数组转换为 XGBoost 的 DMatrix 格式
        dmatrix_data = xgb.DMatrix(input_data_array)

        # 使用模型进行预测
        prediction = model.predict(dmatrix_data)
        prediction_proba = model.predict_proba(dmatrix_data)

        metastasis_proba = float(prediction_proba[0][1]) * 100

        # 绘制极坐标图
        fig, ax = plt.subplots(figsize=(0.7, 0.7), subplot_kw={'projection': 'polar'})
        data = metastasis_proba
        startangle = 90
        x = (data * 2 * pi) / 100
        left = (startangle * pi * 2) / 360

        plt.xticks([])
        plt.yticks([])
        ax.spines.clear()

        ax.barh(1, 360, left=360, height=1, color='#CCCCCC')
        ax.barh(1, x, left=left, height=1, color='#846be0')
        ax.scatter(x + left, 1, s=7, color='#846be0', zorder=2)
        ax.scatter(left,1, s=7, color='#846be0', zorder=2)
        plt.ylim(-3, 3)
        plt.text(0, -2.5, f"{metastasis_proba:.1f}%", ha='center', va='center', fontsize=4)

        img = io.BytesIO()
        plt.savefig(img, dpi=300 , format='PNG', bbox_inches='tight', facecolor='none', edgecolor='none' )
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        response = {
            "metastasis_proba": metastasis_proba,  # 预测概率
            "plot_url": plot_url,  # base64编码的图片
            "prediction": int(prediction[0])  # 预测结果
        }

        return jsonify(response)

    except Exception as e:
        # 捕获所有异常并返回错误消息
        return jsonify({'error': f"预测过程中发生错误: {str(e)}"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
