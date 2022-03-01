# -*- coding: utf-8 -*-

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
from tornado.escape import url_unescape
from uuid import uuid4
import json, ast
import time
import urllib
# import urllib2
import re
import time
import random
import os
import requests
from multiprocessing.dummy import Pool as ThreadPool
from tornado.log import enable_pretty_logging
# enable_pretty_logging()
import logging
import argparse

from test_module import init_model, run_model_with_svg_string, get_word_seq_score


logger = logging.getLogger(__name__)

machine_model = '';


# from tornado.options import define, options
# define("port", default=8999, help = "run on the given port", type = int)


client_file_root_path = "webpages"

# def get_page(url):
#     user_agent_str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36"
#     time.sleep(random.uniform(0,1))
#     return requests.get(url, headers={"Connection":"keep-alive", "User-Agent": user_agent_str}).text


encoder_global = ""
decoder_global = ""
word_map_global = ""
rev_word_map_global = ""

encoder_focus = ""
decoder_focus = ""
word_map_focus = ""
rev_word_map_focus = ""

max_element_number = 100





class getMachineAnswer(tornado.web.RequestHandler):
    # def set_default_headers(self):
    #     self.set_header('Access-Control-Allow-Origin', '*')
    #     self.set_header('Access-Control-Allow-Headers', '*')
    #     self.set_header('Access-Control-Max-Age', 1000)
    #     self.set_header('Content-type', 'application/json')
    #     self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        svg_string = '<svg id="mySvg" width="800" height="350" xmlns="http://www.w3.org/2000/svg"><g transform="translate(80,35)" class="main_canvas"><g class="axis" transform="translate(0,280)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><g class="tick" opacity="1" transform="translate(56.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2010</text></g><g class="tick" opacity="1" transform="translate(144.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2011</text></g><g class="tick" opacity="1" transform="translate(232.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2012</text></g><g class="tick" opacity="1" transform="translate(320.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2013</text></g><g class="tick" opacity="1" transform="translate(408.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2014</text></g><g class="tick" opacity="1" transform="translate(496.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2015</text></g><g class="tick" opacity="1" transform="translate(584.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2016</text></g></g><g class="axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><g class="tick" opacity="1" transform="translate(0,280.5)"><line stroke="currentColor" x2="640"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.0</text></g><g class="tick" opacity="1" transform="translate(0,204.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.5</text></g><g class="tick" opacity="1" transform="translate(0,129.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.0</text></g><g class="tick" opacity="1" transform="translate(0,53.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.5</text></g></g><text transform="translate(-35, 175) rotate(-90)" text-anchor="start" font-size="20px"></text><g><g fill="#fb8072"><rect class="element_0" id="0" o0="2010" c0="F" q0="0.5916307422726574" x="21" y="210" height="70" width="70"></rect><rect class="element_1" id="1" o0="2011" c0="F" q0="0.499292187816756" x="109" y="221" height="59" width="70"></rect><rect class="element_2" id="2" o0="2012" c0="F" q0="0.5833698942964443" x="197" y="211" height="69" width="70"></rect><rect class="element_3" id="3" o0="2013" c0="F" q0="0.580821550862816" x="285" y="211" height="69" width="70"></rect><rect class="element_4" id="4" o0="2014" c0="F" q0="0.5778160391056617" x="373" y="212" height="68" width="70"></rect><rect class="element_5" id="5" o0="2015" c0="F" q0="0.5374138350916083" x="461" y="216" height="64" width="70"></rect><rect class="element_6" id="6" o0="2016" c0="F" q0="0.5190015788311658" x="549" y="219" height="61" width="70"></rect></g><g fill="#d9d9d9"><rect class="element_7" id="7" o0="2010" c0="B" q0="1" x="21" y="92" height="118" width="70"></rect><rect class="element_8" id="8" o0="2011" c0="B" q0="1.0735472169341573" x="109" y="94" height="127" width="70"></rect><rect class="element_9" id="9" o0="2012" c0="B" q0="1.1706944287796168" x="197" y="73" height="138" width="70"></rect><rect class="element_10" id="10" o0="2013" c0="B" q0="1.2575450048813972" x="285" y="63" height="148" width="70"></rect><rect class="element_11" id="11" o0="2014" c0="B" q0="1.3372426823618313" x="373" y="53" height="159" width="70"></rect><rect class="element_12" id="12" o0="2015" c0="B" q0="1.7954980378161547" x="461" y="4" height="212" width="70"></rect><rect class="element_13" id="13" o0="2016" c0="B" q0="1.848192767898499" x="549" y="0" height="219" width="70"></rect></g></g><g transform="translate(640,0)" class="legend-wrap"><g transform="translate(0,0)"><rect width="15.120000000000001" height="15.120000000000001" fill="#fb8072" id="color-0" color-data="#fb8072" custom-id="0" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">F</text></g><g transform="translate(0,16.8)"><rect width="15.120000000000001" height="15.120000000000001" fill="#d9d9d9" id="color-1" color-data="#d9d9d9" custom-id="1" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">B</text></g></g><text class="title" text-anchor="middle" font-size="33.6" x="320" y="-44.8" style="font-family: Oxygen; font-weight: bold; fill: #253039;">THE VALUE</text></g></svg>'
        sentences = get_sentences_from_svg(svg_string)
        self.write(json.dumps({'message': sentences}))
        self.finish()

    def post(self):

        logger.info(url_unescape(self.request.body))
        svg_string = self.get_body_argument('svg_string')
        logger.info(svg_string)

        focus_type = False
        try:
            focus_content = self.get_body_argument('focus')
            if focus_content == "false":
                focus_type = False
            else:
                focus_type = True
            print("from server", focus_content, focus_type)

        finally:
            sentences = get_sentences_from_svg(svg_string, focus_type = focus_type)


        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.write(json.dumps(sentences))
        self.finish()
        # self.finish()

class Application(tornado.web.Application):
    def __init__ (self):
        handlers = [
            (r'/get_caption', getMachineAnswer),
            (r'/(.*)', tornado.web.StaticFileHandler, {'path': client_file_root_path, 'default_filename': 'index.html'}), # fetch client file
            (r'/autocaption/(.*)', tornado.web.StaticFileHandler, {'path': client_file_root_path, 'default_filename': 'index.html'}), # fetch client file
            ]

        settings = {
            'static_path': 'static',
            'debug': True
            }
        tornado.web.Application.__init__(self, handlers, **settings)

def get_sentences_from_svg(svg_string, focus_type = False):

    global encoder_global
    global decoder_global
    global word_map_global
    global rev_word_map_global

    global encoder_focus
    global decoder_focus
    global word_map_focus
    global rev_word_map_focus

    global max_element_number

    if focus_type:
        seqs, alphas, scores, soup, replace_dict, element_number = run_model_with_svg_string(svg_string, encoder_focus, decoder_focus, word_map_focus, rev_word_map_focus, max_element_number = max_element_number, replace_token = True, need_focus = True, focus_mode = "parsed")
        sentences = get_word_seq_score(seqs, rev_word_map_focus, replace_dict, scores)

    else:
        seqs, alphas, scores, soup, replace_dict, element_number = run_model_with_svg_string(svg_string, encoder_global, decoder_global, word_map_global, rev_word_map_global, max_element_number = max_element_number, replace_token = True, need_focus = False)
        sentences = get_word_seq_score(seqs, rev_word_map_global, replace_dict, scores)
   
    return sentences

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--port', default = 9999, type = int, help = "The path to store the setting")
    parser.add_argument('--setting_json', default = 'setting/20200831_setting.json', help = "The path to store the setting")

    args = parser.parse_args()
    args.need_text = True

    print("setting json", args.setting_json)

    with open(args.setting_json) as f:
        setting_json = json.load(f)

    # 从配置文件中读取相应的参数

    model_path_global = setting_json["model_path_global"]
    word_map_path_global = setting_json["word_map_path_global"]

    model_path_focus = setting_json["model_path_focus"]
    word_map_path_focus = setting_json["word_map_path_focus"]

    max_element_number = setting_json["max_element_number"]

    # Load global model
    encoder_global, decoder_global, word_map_global, rev_word_map_global = init_model(model_path_global, word_map_path_global, max_ele_num = max_element_number)
    encoder_focus, decoder_focus, word_map_focus, rev_word_map_focus = init_model(model_path_focus, word_map_path_focus, max_ele_num = max_element_number)

    print(encoder_global)
    print(decoder_global)

    # encoder, decoder, word_map, rev_word_map = init_model(model_path, word_map_path, max_ele_num = max_element_number)

    # svg_string = '<svg id="mySvg" width="800" height="350" xmlns="http://www.w3.org/2000/svg"><g transform="translate(80,35)" class="main_canvas"><g class="axis" transform="translate(0,280)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><g class="tick" opacity="1" transform="translate(56.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2010</text></g><g class="tick" opacity="1" transform="translate(144.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2011</text></g><g class="tick" opacity="1" transform="translate(232.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2012</text></g><g class="tick" opacity="1" transform="translate(320.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2013</text></g><g class="tick" opacity="1" transform="translate(408.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2014</text></g><g class="tick" opacity="1" transform="translate(496.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2015</text></g><g class="tick" opacity="1" transform="translate(584.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2016</text></g></g><g class="axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><g class="tick" opacity="1" transform="translate(0,280.5)"><line stroke="currentColor" x2="640"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.0</text></g><g class="tick" opacity="1" transform="translate(0,204.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.5</text></g><g class="tick" opacity="1" transform="translate(0,129.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.0</text></g><g class="tick" opacity="1" transform="translate(0,53.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.5</text></g></g><text transform="translate(-35, 175) rotate(-90)" text-anchor="start" font-size="20px"></text><g><g fill="#fb8072"><rect class="element_0" id="0" o0="2010" c0="F" q0="0.5916307422726574" x="21" y="210" height="70" width="70"></rect><rect class="element_1" id="1" o0="2011" c0="F" q0="0.499292187816756" x="109" y="221" height="59" width="70"></rect><rect class="element_2" id="2" o0="2012" c0="F" q0="0.5833698942964443" x="197" y="211" height="69" width="70"></rect><rect class="element_3" id="3" o0="2013" c0="F" q0="0.580821550862816" x="285" y="211" height="69" width="70"></rect><rect class="element_4" id="4" o0="2014" c0="F" q0="0.5778160391056617" x="373" y="212" height="68" width="70"></rect><rect class="element_5" id="5" o0="2015" c0="F" q0="0.5374138350916083" x="461" y="216" height="64" width="70"></rect><rect class="element_6" id="6" o0="2016" c0="F" q0="0.5190015788311658" x="549" y="219" height="61" width="70"></rect></g><g fill="#d9d9d9"><rect class="element_7" id="7" o0="2010" c0="B" q0="1" x="21" y="92" height="118" width="70"></rect><rect class="element_8" id="8" o0="2011" c0="B" q0="1.0735472169341573" x="109" y="94" height="127" width="70"></rect><rect class="element_9" id="9" o0="2012" c0="B" q0="1.1706944287796168" x="197" y="73" height="138" width="70"></rect><rect class="element_10" id="10" o0="2013" c0="B" q0="1.2575450048813972" x="285" y="63" height="148" width="70"></rect><rect class="element_11" id="11" o0="2014" c0="B" q0="1.3372426823618313" x="373" y="53" height="159" width="70"></rect><rect class="element_12" id="12" o0="2015" c0="B" q0="1.7954980378161547" x="461" y="4" height="212" width="70"></rect><rect class="element_13" id="13" o0="2016" c0="B" q0="1.848192767898499" x="549" y="0" height="219" width="70"></rect></g></g><g transform="translate(640,0)" class="legend-wrap"><g transform="translate(0,0)"><rect width="15.120000000000001" height="15.120000000000001" fill="#fb8072" id="color-0" color-data="#fb8072" custom-id="0" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">F</text></g><g transform="translate(0,16.8)"><rect width="15.120000000000001" height="15.120000000000001" fill="#d9d9d9" id="color-1" color-data="#d9d9d9" custom-id="1" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">B</text></g></g><text class="title" text-anchor="middle" font-size="33.6" x="320" y="-44.8" style="font-family: Oxygen; font-weight: bold; fill: #253039;">THE VALUE</text></g></svg>'
    
    svg_string = '<svg xmlns="http://www.w3.org/2000/svg" id="img-svg" width="100%" height="100%" viewBox="0 0 485.8125 500" user-select="none" style="width: 100%; height: 100%; user-select: none;"><g transform="translate(80,80)" class="main_canvas" ele-id="1"><g class="axis" transform="translate(0,240)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle" ele-id="2"><g class="tick" opacity="1" transform="translate(56.5,0)" ele-id="3"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;" ele-id="4">ord0</text></g><g class="tick" opacity="1" transform="translate(146.5,0)" ele-id="5"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;" ele-id="6">ord1</text></g><g class="tick" opacity="1" transform="translate(236.5,0)" ele-id="7"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;" ele-id="8">ord2</text></g></g><g class="axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="end" ele-id="9"><g class="tick" opacity="1" transform="translate(0,240.5)" ele-id="10"><line stroke="currentColor" x2="291.4948526542539" ele-id="11"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;" ele-id="12">0</text></g><g class="tick" opacity="1" transform="translate(0,200.5)" ele-id="13"><line stroke="currentColor" x2="291.4948526542539" style="stroke-opacity: 0.3;" ele-id="14"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;" ele-id="15">20</text></g><g class="tick" opacity="1" transform="translate(0,159.5)" ele-id="16"><line stroke="currentColor" x2="291.4948526542539" style="stroke-opacity: 0.3;" ele-id="17"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;" ele-id="18">40</text></g><g class="tick" opacity="1" transform="translate(0,119.5)" ele-id="19"><line stroke="currentColor" x2="291.4948526542539" style="stroke-opacity: 0.3;" ele-id="20"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;" ele-id="21">60</text></g><g class="tick" opacity="1" transform="translate(0,78.5)" ele-id="22"><line stroke="currentColor" x2="291.4948526542539" style="stroke-opacity: 0.3;" ele-id="23"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;" ele-id="24">80</text></g><g class="tick" opacity="1" transform="translate(0,38.5)" ele-id="25"><line stroke="currentColor" x2="291.4948526542539" style="stroke-opacity: 0.3;" ele-id="26"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;" ele-id="27">100</text></g></g><text transform="translate(-35, 200) rotate(-90)" text-anchor="start" font-size="20px" ele-id="28"></text><g class="bars" ele-id="29"><rect fill="#999999" id="0" class="element_0 elements ordinary" x="24" y="72" height="168" width="20.7" rx="1" ry="1" ele-id="30" focusid="yes"></rect><rect fill="#999999" id="1" class="element_1 elements ordinary" x="114" y="12" height="228" width="20.7" rx="1" ry="1" ele-id="31" focusid="yes"></rect><rect fill="#999999" id="2" class="element_2 elements ordinary" x="204" y="0" height="240" width="20.7" rx="1" ry="1" ele-id="32" focusid="yes"></rect><rect fill="#ffff33" id="3" class="element_3 elements ordinary" x="47" y="125" height="115" width="20.7" rx="1" ry="1" ele-id="33"></rect><rect fill="#ffff33" id="4" class="element_4 elements ordinary" x="137" y="137" height="103" width="20.7" rx="1" ry="1" ele-id="34"></rect><rect fill="#ffff33" id="5" class="element_5 elements ordinary" x="227" y="62" height="178" width="20.7" rx="1" ry="1" ele-id="35"></rect><rect fill="#e41a1c" id="6" class="element_6 elements ordinary" x="70" y="94" height="146" width="20.7" rx="1" ry="1" ele-id="36"></rect><rect fill="#e41a1c" id="7" class="element_7 elements ordinary" x="160" y="173" height="67" width="20.7" rx="1" ry="1" ele-id="37"></rect><rect fill="#e41a1c" id="8" class="element_8 elements ordinary" x="250" y="111" height="129" width="20.7" rx="1" ry="1" ele-id="38"></rect></g><g transform="translate(291.4948526542539,0)" class="legend-wrap" ele-id="39"><g transform="translate(0,0)" ele-id="40"><rect width="12.959999999999999" height="12.959999999999999" fill="#999999" id="color-0" color-data="#999999" custom-id="0" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)" ele-id="41"></rect><text x="15.12" y="10.799999999999999" text-anchor="start" font-size="10.799999999999999" ele-id="42">item0</text></g><g transform="translate(0,14.399999999999999)" ele-id="43"><rect width="12.959999999999999" height="12.959999999999999" fill="#ffff33" id="color-1" color-data="#ffff33" custom-id="1" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)" ele-id="44"></rect><text x="15.12" y="10.799999999999999" text-anchor="start" font-size="10.799999999999999" ele-id="45">item1</text></g><g transform="translate(0,28.799999999999997)" ele-id="46"><rect width="12.959999999999999" height="12.959999999999999" fill="#e41a1c" id="color-2" color-data="#e41a1c" custom-id="2" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)" ele-id="47"></rect><text x="15.12" y="10.799999999999999" text-anchor="start" font-size="10.799999999999999" ele-id="48">item2</text></g></g><text class="title" text-anchor="middle" font-size="28.799999999999997" x="145.74742632712696" y="-38.4" style="font-family: Oxygen; font-weight: bold; fill: #253039;" ele-id="49">THE VALUE</text></g></svg>'

    focus_sentences = get_sentences_from_svg(svg_string, focus_type = True)
    global_sentences = get_sentences_from_svg(svg_string, focus_type = False)

    print("focus_sentences", focus_sentences)
    print("global_sentences", global_sentences)

    print('server running at 127.0.0.1:%d ...'%(args.port))

    app = Application()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()


