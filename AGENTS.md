## 项目概述
智慧大棚气象数据预测系统 - 基于 Flask 的 Web 应用，用于展示物联网传感器数据的上传、清洗、聚合以及基于机器学习/时间序列模型的气象数据未来趋势预测。支持温度、湿度、风速、降水量四种气象指标的预测。

## 技术栈
- **语言**：Python 3.12
- **框架**：Flask 3.1.3
- **数据库**：SQLite (SQLAlchemy ORM)
- **机器学习**：Prophet, scikit-learn, XGBoost, statsmodels
- **数据处理**：pandas, numpy, openpyxl
- **可视化**：matplotlib
- **文件格式**：Excel (.xlsx)

## 目录结构
```
/workspace/projects/
├── app.py                 # Flask 主应用入口
├── requirements.txt       # Python 依赖
├── scripts/               # 预览和部署脚本
│   ├── coze-preview-build.sh
│   ├── coze-preview-run.sh
│   ├── coze-deploy-build.sh
│   └── coze-deploy-run.sh
├── templates/             # HTML 模板
├── static/               # 静态资源
├── uploads/              # 用户上传文件 (.xlsx)
├── 温湿度数据csv/        # 转换后的 CSV 数据
└── 预测结果/             # 预测输出
```

## 关键入口 / 核心模块
- `app.py`: Flask 应用主入口，监听 0.0.0.0:5000
- 核心路由：/ (首页), /predict_center (预测中心), /admin/* (管理后台)
- `convert_to_csv.py`: xlsx 文件转换模块

## xlsx 数据格式
上传的 xlsx 文件应包含以下列：
- **收集时间**: 日期时间
- **大气温度** → 预测目标: 温度
- **大气湿度** → 预测目标: 湿度
- **风速** → 预测目标: 风速
- **降雨量** → 预测目标: 降水量

## 支持的预测类型
1. **温度** (℃) - 范围: -20 ~ 50
2. **湿度** (%) - 范围: 0 ~ 100
3. **风速** (m/s) - 范围: 0 ~ 30
4. **降水量** (mm) - 范围: 0 ~ 100

## 运行与预览
```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发预览
python app.py  # 默认端口 5000

# 默认管理员账号
# 用户名: admin
# 密码: admin123
```

## 项目初始化信息
- **初始化日期**: 2025-04-29
- **sub_id**: 117fd3d4
- **project_type**: web
- **preview_enable**: enabled
- **deploy.kind**: service
- **deploy.flavor**: web

## 用户偏好与长期约束
- Flask 应用已修改为监听 0.0.0.0:5000（app.py 中的 app.run 参数）
- 使用脚本进行依赖安装和端口清理，确保幂等性
- Python 运行时版本需 >= 3.10（建议 3.12）
