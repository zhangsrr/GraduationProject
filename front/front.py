from flask import Blueprint, render_template

front_bp = Blueprint('front', __name__)


# 主页面显示
@front_bp.route('/')
@front_bp.route('/MainMenu')
def UIMain():
    return render_template('/UIMain.html')


@front_bp.route('/SearchPRB')
def UIsearchPRB():
    netname = get_all_PRB_netname()
    colName = get_all_PRB_attr()
    return render_template('UISearchPRB.html', net_name=netname, attr_list=colName)
