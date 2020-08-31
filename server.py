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


encoder = ""
decoder = ""
word_map = ""
rev_word_map = ""
args = ""


class getMachineAnswer(tornado.web.RequestHandler):
    # def set_default_headers(self):
    #     self.set_header('Access-Control-Allow-Origin', '*')
    #     self.set_header('Access-Control-Allow-Headers', '*')
    #     self.set_header('Access-Control-Max-Age', 1000)
    #     self.set_header('Content-type', 'application/json')
    #     self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.write(json.dumps({'message': sentences}))
        self.finish()

    def post(self):

        logger.info(url_unescape(self.request.body))
        svg_string = self.get_body_argument('svg_string')
        logger.info(svg_string)

        sentences = get_sentences_from_svg(svg_string)
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.write(json.dumps(sentences))
        self.finish()
        # self.finish()

class Application(tornado.web.Application):
    def __init__ (self):
        handlers = [
            (r'/autocaption/get_machine_answer', getMachineAnswer),
            (r'/(.*)', tornado.web.StaticFileHandler, {'path': client_file_root_path, 'default_filename': 'index.html'}), # fetch client file
            (r'/autocaption/(.*)', tornado.web.StaticFileHandler, {'path': client_file_root_path, 'default_filename': 'index.html'}), # fetch client file
            ]

        settings = {
            'static_path': 'static',
            'debug': True
            }
        tornado.web.Application.__init__(self, handlers, **settings)

def get_sentences_from_svg(svg_string):
    global encoder
    global decoder
    global word_map
    global rev_word_map
    global args
    seqs, alphas, scores, soup, replace_dict, element_number = run_model_with_svg_string(svg_string, encoder, decoder, word_map, rev_word_map, max_element_number = max_element_number, replace_token = args.replace_token, need_focus = args.need_focus, focus = args.focus)
    sentences = get_word_seq_score(seqs, rev_word_map, replace_dict, scores)
    return sentences

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', default = "", help='path to image')
    parser.add_argument('--model', '-m', default = "checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar",  help='path to model')
    parser.add_argument('--word_map', '-wm', default = "data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json", help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--image_type', type=str, default = 'svg', help='image type as input')
    parser.add_argument('--need_text', action='store_true', help="decide whether need text")
    parser.add_argument('--max_element_number', '-e', default=100, type=int, help='maximum element number')
    parser.add_argument('--port', '-p', default=9999, type=int, help='maximum element number')
    parser.add_argument('--replace_token', action = "store_true", help="replace token")
    parser.add_argument('--result_file', default = "tmp.json", help = "temperal file to store the results")
    parser.add_argument('--need_focus', action = "store_true", help = "Using focus as input")
    parser.add_argument('--focus', default = '0,1,2', help = "The array of focused id of the chart")

    args = parser.parse_args()
    args.need_text = True

    max_element_number = args.max_element_number

    model_path = args.model
    word_map_path = args.word_map
    image_path = args.img 


    encoder, decoder, word_map, rev_word_map = init_model(model_path, word_map_path, max_ele_num = max_element_number)



    init_model(args.model, args.word_map, max_ele_num = args.max_element_number)
    # svg_string = '<svg id="mySvg" width="800" height="350" xmlns="http://www.w3.org/2000/svg"><g transform="translate(80,35)" class="main_canvas"><g class="axis" transform="translate(0,280)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><g class="tick" opacity="1" transform="translate(56.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2010</text></g><g class="tick" opacity="1" transform="translate(144.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2011</text></g><g class="tick" opacity="1" transform="translate(232.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2012</text></g><g class="tick" opacity="1" transform="translate(320.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2013</text></g><g class="tick" opacity="1" transform="translate(408.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2014</text></g><g class="tick" opacity="1" transform="translate(496.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2015</text></g><g class="tick" opacity="1" transform="translate(584.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2016</text></g></g><g class="axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><g class="tick" opacity="1" transform="translate(0,280.5)"><line stroke="currentColor" x2="640"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.0</text></g><g class="tick" opacity="1" transform="translate(0,204.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.5</text></g><g class="tick" opacity="1" transform="translate(0,129.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.0</text></g><g class="tick" opacity="1" transform="translate(0,53.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.5</text></g></g><text transform="translate(-35, 175) rotate(-90)" text-anchor="start" font-size="20px"></text><g><g fill="#fb8072"><rect class="element_0" id="0" o0="2010" c0="F" q0="0.5916307422726574" x="21" y="210" height="70" width="70"></rect><rect class="element_1" id="1" o0="2011" c0="F" q0="0.499292187816756" x="109" y="221" height="59" width="70"></rect><rect class="element_2" id="2" o0="2012" c0="F" q0="0.5833698942964443" x="197" y="211" height="69" width="70"></rect><rect class="element_3" id="3" o0="2013" c0="F" q0="0.580821550862816" x="285" y="211" height="69" width="70"></rect><rect class="element_4" id="4" o0="2014" c0="F" q0="0.5778160391056617" x="373" y="212" height="68" width="70"></rect><rect class="element_5" id="5" o0="2015" c0="F" q0="0.5374138350916083" x="461" y="216" height="64" width="70"></rect><rect class="element_6" id="6" o0="2016" c0="F" q0="0.5190015788311658" x="549" y="219" height="61" width="70"></rect></g><g fill="#d9d9d9"><rect class="element_7" id="7" o0="2010" c0="B" q0="1" x="21" y="92" height="118" width="70"></rect><rect class="element_8" id="8" o0="2011" c0="B" q0="1.0735472169341573" x="109" y="94" height="127" width="70"></rect><rect class="element_9" id="9" o0="2012" c0="B" q0="1.1706944287796168" x="197" y="73" height="138" width="70"></rect><rect class="element_10" id="10" o0="2013" c0="B" q0="1.2575450048813972" x="285" y="63" height="148" width="70"></rect><rect class="element_11" id="11" o0="2014" c0="B" q0="1.3372426823618313" x="373" y="53" height="159" width="70"></rect><rect class="element_12" id="12" o0="2015" c0="B" q0="1.7954980378161547" x="461" y="4" height="212" width="70"></rect><rect class="element_13" id="13" o0="2016" c0="B" q0="1.848192767898499" x="549" y="0" height="219" width="70"></rect></g></g><g transform="translate(640,0)" class="legend-wrap"><g transform="translate(0,0)"><rect width="15.120000000000001" height="15.120000000000001" fill="#fb8072" id="color-0" color-data="#fb8072" custom-id="0" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">F</text></g><g transform="translate(0,16.8)"><rect width="15.120000000000001" height="15.120000000000001" fill="#d9d9d9" id="color-1" color-data="#d9d9d9" custom-id="1" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">B</text></g></g><text class="title" text-anchor="middle" font-size="33.6" x="320" y="-44.8" style="font-family: Oxygen; font-weight: bold; fill: #253039;">THE VALUE</text></g></svg>'
    
    # seqs, alphas, scores, soup, replace_dict, element_number = run_model_with_svg_string(svg_string, encoder, decoder, word_map, rev_word_map, max_element_number = max_element_number, replace_token = args.replace_token, need_focus = args.need_focus, focus = args.focus)

    # sentences = get_word_seq_score(seqs, rev_word_map, replace_dict, scores)

    print('server running at 127.0.0.1:%d ...'%(args.port))

    app = Application()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()


