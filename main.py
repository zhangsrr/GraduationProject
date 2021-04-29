from flask import Flask
import os

from front.front import front_bp as front_blueprint

os.environ["FLASK_ENV"] = "development"  # 设置运行模式

app = Flask(__name__)

app.register_blueprint(front_blueprint)  # 前端页面显示接口

if __name__ == '__main__':
    import datetime
    print(datetime.datetime.now())
    app.run(host='127.0.0.1', port=5000, debug=True)
