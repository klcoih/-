## 项目概述
智慧大棚温湿度预测系统 - 基于 Flask 的 Web 应用，用于展示物联网传感器数据的上传、清洗、聚合以及基于机器学习/时间序列模型的温湿度未来趋势预测。

## 技术栈
- **语言**：Python 3.12
- **框架**：Flask 3.1.3
- **数据库**：SQLite (SQLAlchemy ORM)
- **机器学习**：Prophet, scikit-learn, XGBoost, statsmodels
- **数据处理**：pandas, numpy
- **可视化**：matplotlib

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
├── uploads/              # 用户上传文件
├── 温湿度数据csv/        # 转换后的 CSV 数据
└── 预测结果/             # 预测输出
```

## 关键入口 / 核心模块
- `app.py`: Flask 应用主入口，监听 0.0.0.0:5000
- 核心路由：/ (首页), /predict (预测), /admin/* (管理后台)

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
