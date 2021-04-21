from flask import Flask
import os

from front.front import front_bp as front_blueprint

# from backend.im_ex_port import im_ex_port as im_ex_port_bp
# from backend.info_query import info_query as info_query_bp
# from backend.c2i_triple_tuple import backend_c2i_triple_tuple as c2i_tri_blueprint
# from backend.user import backend_user as user_bp
os.environ["FLASK_ENV"] = "development" #设置运行模式

app = Flask(__name__)

app.register_blueprint(front_blueprint) # 前端页面显示接口


# 待添加数据导入导出接口
# app.register_blueprint(im_ex_port_bp)
#
# app.register_blueprint(info_query_bp)  # 数据查询的后端接口
# app.register_blueprint(c2i_tri_blueprint)  # C2I和重叠覆盖干扰三元组的后端接口


import datetime
if __name__ == '__main__':
    print(datetime.datetime.now())
    app.run(host='127.0.0.1', port=5000, debug=True)
