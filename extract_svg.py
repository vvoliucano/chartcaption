import json
import bs4
# import os
import numpy
import re
# from svgpathtools import parse_path, Line, disvg
import copy
import svgpathtools

from svg.path import parse_path
from svg.path.path import Line
# def get_attr_by_style(element):
    #
def is_number(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        pass
    return False

def get_rect_attr(rect, attr, default_value):
    if attr in rect.keys():
        return rect[attr]
    else:
        return default_value

def try_convert_number(s):
    try:
        number = float(s.replace(",", ""))
        return number 
    except ValueError:
        pass
    return s

def parse_fill(fill):
    # print(fill)
    if type(fill)==list and len(fill)==3:
        return fill
    elif fill == "currentColor":
        return [0, 0, 0]
    elif fill[0] != "#":
        print(f"I cannot handle this color {fill}")
        return [0, 0, 0]
    elif len(fill) == 7:
        r = int(fill[1:3], 16)/255.0;
        g = int(fill[3:5], 16)/255.0;
        b = int(fill[5:7], 16)/255.0;
        # print(r, g, b)
        return [r, g, b]
    elif len(fill) == 4:
        r = int(fill[1:2], 16)/15.0;
        g = int(fill[2:3], 16)/15.0;
        b = int(fill[3:4], 16)/15.0;
        # print(r, g, b)
        return [r, g, b]
    return [0,0,0]

def get_attr(element, attr, default_value = ""):
    # 这是一个补丁
    if attr == "fill":
        if not element.has_attr(attr):
            for node in element.parents:
                if node.name == "g":
                    if node.has_attr(attr):
                        return node[attr]
        else:
            return parse_fill(element[attr])

    if attr == "text-anchor":
        # print(element)
        # print("元素有这个吗？", element.has_attr(attr))
        if element.has_attr(attr):
            return element[attr]
        else:
            for node in element.parents:
                if node.name == "g":
                    if node.has_attr(attr):
                        return node[attr]
        return "start"


    elif element.has_attr(attr):
        if attr == "width" or attr == "height":
            if element[attr].endswith("%"):
                return default_value
        elif attr == "r":
            return re.sub("[a-z]", "", element[attr])
        elif attr == "font-size":
            font_size_value = element[attr]
            font_size_value = font_size_value.replace("px", '')
            if font_size_value.endswith("em"):
                relative_value = float(font_size_value.replace("em", "")) * 12
                return relative_value
            # print(font_size_value)
            return font_size_value

        return element[attr]
    else:
        return default_value

def parse_transform(element):
    transform = get_attr(element, "transform", "translate(0,0)")
    x = transform.split("(")[1].split(",")[0]
    y = transform.split(",")[1].split(")")[0]
    x = float(x)
    y = float(y)

    return x, y

def get_font_size(element):
    if element.name == "text":
        font_size = float(get_attr(element, "font-size", 12))
    
        return font_size

def get_translate(element):
    if not element.has_attr("transform"):
        return 0,0
    transform = element['transform']
    if not transform.startswith("translate("):
        return 0,0
    xy = transform.replace("translate(", "").replace(")", "").split(",")

    x = float(xy[0])
    y = float(xy[1])
    # print(f"deal with transform: {x} {y}")
    return x,y


def get_position(element, is_bbox = False):
    # print(element.name)
    if not is_bbox:
        if element.name == "rect":
            x = float(get_attr(element, "x", 0))
            y = float(get_attr(element, "y", 0))
            dx, dy = get_translate(element)
            x = x + dx
            y = y + dy
        elif element.name == "circle":
            x = float(get_attr(element, "cx", 0))
            y = float(get_attr(element, "cy", 0))
        elif element.name == "text":
            x = float(get_attr(element, "x", 0))
            y = float(get_attr(element, "y", 0))
        elif element.name == "path":
            x, y = parse_transform(element)
        elif element.name == "line":
            x = 0
            y = 0
        else:
            print("can not handle current element type: ", element.name)
    else:
        x = float(get_attr(element, "bbox_x", 0))
        y = float(get_attr(element, "bbox_y", 0))



    # print(f"Now, x: {x}, y: {y}")
    for parent in element.parents:
        if parent.name == "svg":
            break;
        if parent.name == "g":
            add_x, add_y = parse_transform(parent)
            x = x + add_x
            y = y + add_y

    # print(f"Now, x: {x}, y: {y}")
    return x, y


def get_rectangles(soup):
    rects = soup.select("rect")
    return rects

def path_a_line_seg():
    width = float(get_attr(rect, "width", 0))
    height = float(get_attr(rect, "height", 0))
    opacity = float(get_attr(rect, "opacity", 1))
    color  = parse_fill(get_attr(rect, "fill", "#000"))


    # print(get_attr(rect, "fill", "#000"))
    # for debug
    value = float(get_attr(rect, "q0", 0))
    x, y = get_position(rect)
    left = x
    right = x + width
    up = y
    down = y + height
    rect_attr = {
        "type": "rect",
        "origin": rect,
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "value": value,
        "fill": color,
        "opacity": opacity,
        "x": x,
        "y": y,
        "up": up,
        "down": down,
        "text": ""}
    return rect_attr

def get_important_rects(rects, dim, array):
    important_rects = []
    other_rects = []
    sorted_array = sorted(array.items(), key = lambda item:item[1], reverse = True)
    common_value = sorted_array[0][0]
    if common_value == 0:
        common_value = sorted_array[1][0]
    for rect in rects:
        if rect[dim] == common_value:
            important_rects.append(rect)
        elif rect["width"] > 0 and rect["height"] > 0 and rect["opacity"] > 0:
            other_rects.append(rect)
    return important_rects, other_rects

def parse_a_path(path):
    # pathObj = parse_path(path["d"])
    # for parent in path.parents:
    #     if parent.name == "svg":
    #         break
    #     if parent.name == "g":
    #         add_x, add_y = parse_transform(parent)
    # path_attr = {
    #     "origin": path,
    #     "pathObj": pathObj,
    #     "sx": pathObj[0].start.real,
    #     "sy": pathObj[0].start.imag,
    #     "ex": pathObj[-1].end.real,
    #     "ey": pathObj[-1].end.imag,
    #     "rx": add_x,
    #     "ry": add_y,
    #     "color": get_attr(path, "stroke", "#000"),
    # }
    # print("DEBUG", path_attr)
    return path_attr

def parse_a_circle(circle):
    radius = float(get_attr(circle, "r", 0))
    opacity = float(get_attr(circle, "opacity", 1))
    color = get_attr(circle, "fill", "#000")
    x, y = get_position(circle)
    left = x - radius
    right = x + radius
    up = y - radius
    down = y + radius
    width = 2 * radius
    height = 2 * radius
    circle_attr = {
        "origin"    : circle,
        "width"     : width,
        "height"    : height,
        "left"      : left,
        "right"     : right,
        "fill"      : color,
        "opacity"   : opacity,
        "x"         : x,
        "y"         : y,
        "up"        : up,
        "down"      : down,
        "r"         : radius
    }
    return circle_attr



def parse_a_rect(rect):
    width = float(get_attr(rect, "width", 0))
    height = float(get_attr(rect, "height", 0))
    opacity = float(get_attr(rect, "opacity", 1))
    color  = parse_fill(get_attr(rect, "fill", "#000"))


    # print(get_attr(rect, "fill", "#000"))
    # for debug
    value = float(get_attr(rect, "q0", 0))
    x, y = get_position(rect)
    left = x
    right = x + width
    up = y
    down = y + height
    rect_attr = {
        "type": "rect",
        "origin": rect,
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "value": value,
        "fill": color,
        "opacity": opacity,
        "x": x,
        "y": y,
        "up": up,
        "down": down,
        "text": ""}
    return rect_attr

def parse_a_text_visual(text):
    value = float(get_attr(text, "q0", 0)) #走个形式，省点代码，其实没有卵用
    color  = parse_fill(get_attr(text, "fill", "#000"))
    opacity = float(get_attr(text, "opacity", 1))
    x, y = get_position(text)
    font_size = get_font_size(text)
    # print("text", text)
    # print("text string", text.string)
    if text.string == None:
        content = ""
    else:
        content = text.string.replace("\n", "").strip().lower()

    # 此处只是用到非常虚弱的假设，width = font-size * 字母的数目
    # 而，height = font-size, 当然了在不同的字体中会有所不同，

    height = font_size
    width = font_size * len(content)
    text_anchor = get_attr(text, "text-anchor", "start")
    # print(text_anchor)
    if text_anchor == "start":
        x = x;
    elif text_anchor == "middle":
        x = x - width/2 
    else:
        x = x - width

    left = x
    right = x + width
    up = y - height
    down = y
    rect_attr = {
        "type": "text",
        "origin": text,
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "value": value,
        "fill": color,
        "opacity": opacity,
        "x": x,
        "y": y,
        "up": up,
        "down": down,
        "text": content
        }
    return rect_attr


    # bbox_w = float(get_attr(text, "bbox_w", 0))
    # bbox_h = float(get_attr(text, "bbox_h", 0))

    # # print("bbox content", bbox_x, bbox_y, bbox_w, bbox_h)
    
    # # print(content)
    # return_content = {"x": x, "y": y, "content": content, "orgin":text, "font_size": font_size, "text_anchor": text_anchor, "bbox_x": bbox_x, "bbox_y": bbox_y, "bbox_w": bbox_w, "bbox_h": bbox_h}
    # # print("text content", return_content)
    # return return_content

def uniform_important_circle(data):
    q0 = [x['q0'] for x in data['data_array']]
    q1 = [x['q1'] for x in data['data_array']]
    min0 = min(q0)
    min1 = min(q1)
    max0 = max(q0)
    max1 = max(q1)
    circles = copy.deepcopy(data["data_array"])
    for c in circles:
        c['q0'] = (c['q0']-min0)/(max0-min0)
        c['q1'] = (c['q1']-min1)/(max1-min1)
    return circles

def uniform_important_datapoint(data):
    o0 = data["o0"]
    dps = copy.deepcopy(data["data_array"])
    q0 = [dp["q0"] for dp in dps]
    max_value = max(q0)
    min_value = min(q0)
    max_o = max(o0)
    min_o = min(o0)
    for i in range(len(dps)):
        dps[i]["q0"] = (dps[i]["q0"]-min_value)/(max_value-min_value)
        dps[i]["point_x"] = (o0[dps[i]["o0"]] - min_o)/(max_o-min_o)
        dps[i]["point_y"] = dps[i]["q0"]
    return dps

def uniform_important_elements(important_rects):
    top_most = min([rect['up'] for rect in important_rects])
    bottom_most = max([rect['down'] for rect in important_rects])
    left_most = min([rect['left'] for rect in important_rects])
    right_most = max([rect['right'] for rect in important_rects])

    total_width = right_most - left_most
    total_height = bottom_most - top_most
    max_value = max([rect["value"] for rect in important_rects])
    # print(max_value)
    uniform_elements = []
    for rect in important_rects:
        rect["left"] = (rect["left"] - left_most) / total_width
        rect["right"] = (rect["right"] - left_most) / total_width
        rect["up"] = (rect["up"] - top_most) / total_height
        rect["down"] = (rect["down"] - top_most) / total_height
        rect["width"] = rect["width"] / total_width
        rect["height"] = rect["height"] / total_height
        if "value" in rect and max_value != 0:
            rect["value"] = rect["value"] / max_value
        uniform_elements.append(rect)
    return uniform_elements

def get_text_bbox(text_element):

    text_anchor = text_element["text_anchor"]
    content = text_element["content"]
    length = len(text_element["content"])
    font_size = text_element["font_size"]
    width = text_element["bbox_w"]
    height = text_element["bbox_h"]
    x = text_element['bbox_x']
    y = text_element['bbox_y']

    text_bbox = {}
    text_bbox["x"] = x 
    text_bbox["y"] = y 
    text_bbox["w"] = width 
    text_bbox["h"] = height 

    text_bbox["content"] = try_convert_number(content)

    return text_bbox

    # if text_anchor == "start":


def get_text_group(original_text_group, texts_attr, is_legend = False):
    array = []
    if isinstance(original_text_group["x"], list):
        array = original_text_group["x"]
    elif isinstance(original_text_group["y"], list):
        array = original_text_group["y"]



    text_array = [texts_attr[item["text_id"]] for item in array]
    text_bbox = [get_text_bbox(item) for item in text_array]
    if is_legend:
        for i, text in enumerate(text_bbox):
            text["legend_id"] = i
    return text_bbox

def get_text_information(X_axis, Y_axis, legend, texts_attr):
    xAxis_text = get_text_group(X_axis, texts_attr)
    yAxis_text = get_text_group(Y_axis, texts_attr)
    legend_text = get_text_group(legend, texts_attr, is_legend = True)

    # print("formal_x_axis_array", xAxis_text)
    # print("formal_y_axis_array", yAxis_text)
    # print("formal_legend_axis_array", legend_text)
    text_collection = {}
    text_collection['xAxis'] = {}
    text_collection['xAxis']["text"] = xAxis_text
    text_collection['yAxis'] = {}
    text_collection['yAxis']['text'] = yAxis_text 
    text_collection['legend'] = {}
    text_collection['legend']['text'] = legend_text
    text_collection['element'] = []

    return text_collection

def parse_line_seg(soup):
    paths = soup.select('path')
    lines = soup.select("line")

    line_segs = []
    # print(paths)
    for path in paths:
        path_string = path["d"]
        relative_position = get_position(path)
        # print("position", relative_position)
        color = parse_fill(path["stroke"])
        # print("color", color)

        # print(path_string)
        segs = parse_path(path_string)
        for seg in segs:
            if isinstance(seg, Line):
                x0 = seg.start.real + relative_position[0]
                y0 = seg.start.imag + relative_position[1]
                x1 = seg.end.real + relative_position[0]
                y1 = seg.end.imag + relative_position[1]
                parsed_line_seg = packed_formated_line_seg(color, x0, x1, y0, y1, path)
                line_segs.append(parsed_line_seg)

                # x0 = seg.start.real
                # y0 = seg.start.imag
                # x1 = seg.end.real
                # y1 = seg.end.imag
    for line in lines:
        color = parse_fill(line["stroke"])
        x, y = get_position(line)
        # print("the position: ", x, y)
        # print( get_attr(line, "y1", 0))
        x0 = float(get_attr(line, "x1", 0)) + x
        x1 = float(get_attr(line, "x2", 0)) + x
        y0 = float(get_attr(line, "y1", 0)) + y
        y1 = float(get_attr(line, "y2", 0)) + y

        parsed_line_seg = packed_formated_line_seg(color, x0, x1, y0, y1, line)
        line_segs.append(parsed_line_seg)




                # print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))
        # print(segs)
    return line_segs
    # print(line_segs)

def packed_formated_line_seg(color, x0, x1, y0, y1, path):
    
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    opacity = 1
    value = 0
    left = x0 
    right = x1
    up = y0
    down = y1
    line_seg_attr = {
        "type": "line",
        "origin": path,
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "value": value,
        "fill": color,
        "opacity": opacity,
        "x": min(x0, x1),
        "y": min(y0, y1),
        "up": up,
        "down": down,
        "text": ""
        }
    return line_seg_attr


def parse_unknown_svg_visual_elements(svg_string, need_data_soup = False, need_text = False):
    # need_text = True
    # print("this need_text is ", need_text)
    soup = bs4.BeautifulSoup(svg_string, "html5lib")
    svg = soup.select("svg")
    rects = soup.select("rect")
    rects_attr = [parse_a_rect(rect) for rect in rects]

    line_segments = parse_line_seg(soup)

    texts = soup.select("text")
    newtexts = []
    for text in texts:
        if text.has_attr("transform") and "-90" in text["transform"]:
            continue
        else:
            newtexts.append(text)

    texts = newtexts
    # print("texts length", len(texts))
    # print("text", texts)
    texts_attr = [parse_a_text_visual(text) for text in texts]
    # print("texts_attr", texts_attr)

    # Add text
    if need_text:
        rects_attr.extend(texts_attr)

    # Add line
    rects_attr.extend(line_segments)


    for i, rect in enumerate(rects_attr):
        rect["origin"]["caption_id"] = str(i)


    # print("texts_attr", texts_attr)

    text_information = [rect['text'] for rect in rects_attr]
    # print("text information", text_information)

    important_rects = uniform_important_elements(rects_attr)
    data = {}

    if need_text:
        return important_rects, data, soup, text_information

    return important_rects, data, soup

def getCircleList(data):
    cate_choice_number = 15
    type = [0, 1, 0]
    position = [2*data['r'], 2*data['r'], data['q0'], data['q1'], data['q0'], data['q1']]
    color = [0, 0, 0, 1]
    quantity = [data['q0'], data['q1']]
    co= [0 for i in range(4*cate_choice_number)]
    list = type + position + color + quantity + co
    return list

    # width/svg_width, height/svg_height, left/svg_width, right/svg_width, up/svg_height, down/svg_height]
def getDataPointList(data):
    cate_choice_number = 15
    eps = 1e-5
    type = [0, 0, 1]
    position = [eps, eps, data['point_x'], data['point_y'], data['point_x'], data['point_y']]
    color = data['color']
    opacity = [1] #!!!!!!!!!TODO
    quantity = [data['q0'], 0]
    cate0_array = [0 for i in range(cate_choice_number)]
    cate1_array = [0 for i in range(cate_choice_number)]
    ordi0_array = [0 for i in range(cate_choice_number)]
    ordi1_array = [0 for i in range(cate_choice_number)]
    cate0_choice = data['c0']
    ordi0_choice = data['o0']
    cate0_array[cate0_choice] = 1
    ordi0_array[ordi0_choice] = 1
    list = type + position + color + opacity + quantity + cate0_array + cate1_array + ordi0_array
    return list

def parse_svg_string(svg_string, min_element_num = 7, simple = False, need_text = False, need_focus = False):
    
    if need_text:
        important_rects, data, soup, text = parse_unknown_svg_visual_elements(svg_string, need_text = need_text)
    else:
        important_rects, data, soup = parse_unknown_svg_visual_elements(svg_string, need_text)

    # print(important_rects)
    # verify_parsed_results(important_rects)

    is_focus = [get_attr(important_rects[i]["origin"], "focusid", "no") for i in range(len(important_rects))]
    # print("is_focused: ", is_focus)

    focus_array = [i for i in range(len(is_focus)) if is_focus[i] == "yes"]

    elements = []

    for rect in important_rects:
        list = get_rect_list_visual(rect)
        elements.append(list)
    if len(important_rects) < min_element_num:
        for i in range(len(important_rects), min_element_num):
            elements.append([0 for i in range(len(elements[0]))])
        if need_text:
            text.extend(["<pad>" for i in range(min_element_num - len(text))])
            # print("text after uniform", text)
    # print("I want to see the important rects")
    # print("element number", len(elements[0]))
    # print(important_rects)
    # for i, rect in enumerate(important_rects):
    #     rect["id"] = i
    id_array = [i for i in range(len(important_rects))]
    # print(id_array)
    if sum(id_array) == - len(id_array):
        id_array = [i for i in range(len(id_array))]
    # print(f"The id array is {id_array}")

    return_list = [numpy.asarray(elements), id_array, soup]

    if need_text:
        return_list.append(text) 

    if need_focus:
        return_list.append(focus_array)

    return tuple(return_list)


def get_rect_list_visual(rect):
    this_type = [1, 0, 0, 0]
    if rect["type"] == "rect":
        this_type = [1, 0, 0, 0]
    elif rect["type"] == "text":
        this_type = [0, 0, 0, 1]
    elif rect["type"] == "line":
        this_type = [0, 1, 0, 0]

    position = [rect['width'], rect['height'], rect['left'], rect['right'], rect['up'], rect['down']]
    color = rect['fill']
    opacity = [rect['opacity']]
    quantity = [get_rect_attr(rect, 'q0', 0), get_rect_attr(rect, 'q1', 0)]

    list = this_type + position + color + opacity
    # print(f'The attribute of each rectangle is {len(list)}')
    return list

def verify_parsed_results(important_rects):
    from PIL import Image, ImageDraw 

    width = 300
    height = 300

    im = Image.new("RGBA", (width, height), (255,255,255,255))
    draw = ImageDraw.Draw(im)


    for rect in important_rects:
        if rect["type"] == "rect" or rect["type"] == "text":
            draw.rectangle(((rect["left"] * width, rect["up"] * width), (rect["right"] * height, rect["down"] * height)), fill="black")
            draw.text((rect["left"] * width, rect["up"] * width), rect["text"], fill="white")
        else:
            print(rect)
            color = (int(255 * rect["fill"][0]), int(255 * rect["fill"][1]), int(255 * rect["fill"][2]))
            draw.line(((rect["left"] * width, rect["up"] * width), (rect["right"] * height, rect["down"] * height)), fill=color )



    # draw.line((0, 0) + im.size, fill=128)
    # draw.line((0, im.size[1], im.size[0], 0), fill=128)
    im.show()

if __name__ == "__main__":
    # with open("../user_data/cq_liucan_20180827/cq_liucan_2018_08_27_22_26_43_53308.json") as f:
    #     svg_string = json.load(f)["svg_string"]
    # print(svg_string)

    # open_json_file("/Users/tsunmac/vis/projects/autocaption/AutoCaption/user_data/20180918_full_ocq_rule/ocq_super_rule_ocq_web_2018_09_18_12_29_28_1473584.json")
    
    # line chart
    svg_string = '<svg id="mySvg" viewBox="0 0 581.2287479831438 400" preserveAspectRatio="xMidYMid meet" width="581.2287479831438" height="500" xmlns="http://www.w3.org/2000/svg"><g transform="translate(80,80)" class="main_canvas"><g transform="translate(0,320)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><path class="domain" stroke="currentColor" d="M0.5,6V0.5H465.482998386515V6"></path><g class="tick" opacity="1" transform="translate(0.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">0.0</text></g><g class="tick" opacity="1" transform="translate(116.74574959662876,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">0.5</text></g><g class="tick" opacity="1" transform="translate(232.9914991932575,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">1.0</text></g><g class="tick" opacity="1" transform="translate(349.2372487898863,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">1.5</text></g><g class="tick" opacity="1" transform="translate(465.482998386515,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">2.0</text></g></g><text x="232.4914991932575" text-anchor="middle" font-size="23.249149919325752">The Price</text><g fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><path class="domain" stroke="currentColor" d="M-6,320.5H0.5V0.5H-6"></path><g class="tick" opacity="1" transform="translate(0,320.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">0</text></g><g class="tick" opacity="1" transform="translate(0,288.3714859437751)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">10</text></g><g class="tick" opacity="1" transform="translate(0,256.2429718875502)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">20</text></g><g class="tick" opacity="1" transform="translate(0,224.1144578313253)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">30</text></g><g class="tick" opacity="1" transform="translate(0,191.9859437751004)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">40</text></g><g class="tick" opacity="1" transform="translate(0,159.8574297188755)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">50</text></g><g class="tick" opacity="1" transform="translate(0,127.72891566265059)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">60</text></g><g class="tick" opacity="1" transform="translate(0,95.60040160642568)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">70</text></g><g class="tick" opacity="1" transform="translate(0,63.471887550200805)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">80</text></g><g class="tick" opacity="1" transform="translate(0,31.343373493975896)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">90</text></g></g><g id="content"><path d="M0,210.76305220883535L232.4914991932575,125.61218219071765L464.982998386515,53.33333333333332" stroke="#e78ac3" transform="translate(100, 20)" style="stroke-width: 4; fill: none;"></path><path d="M0,114.33651979351421L232.4914991932575,133.96057583167877L464.982998386515,54.56002498101807" stroke="#b3b3b3" style="stroke-width: 4; fill: none;"></path><path d="M0,197.91164658634534L232.4914991932575,230.0401606425703L464.982998386515,230.0401606425703" stroke="#a6d854" style="stroke-width: 4; fill: none;"></path></g><g transform="translate(464.982998386515,0)" class="legend-wrap"><g transform="translate(0,0)"><line x1="0" x2="17.28" y1="8.64" y2="8.64" stroke="#e78ac3" id="color-0" stroke-width="4"></line><text x="20.16" y="14.399999999999999" text-anchor="start" font-size="14.399999999999999">item0</text></g><g transform="translate(0,19.2)"><line x1="0" x2="17.28" y1="8.64" y2="8.64" stroke="#b3b3b3" id="color-1" stroke-width="4"></line><text x="20.16" y="14.399999999999999" text-anchor="start" font-size="14.399999999999999">item1</text></g><g transform="translate(0,38.4)"><line x1="0" x2="17.28" y1="8.64" y2="8.64" stroke="#a6d854" id="color-2" stroke-width="4"></line><text x="20.16" y="14.399999999999999" text-anchor="start" font-size="14.399999999999999">item2</text></g></g></g></svg>'

    # bar chart
    # svg_string = '<svg id="mySvg" viewBox="0 0 609.3764369175353 400" preserveAspectRatio="xMidYMid meet" height="500" width="609.3764369175353" xmlns="http://www.w3.org/2000/svg"><g transform="translate(80,80)" class="main_canvas"><g class="brush"></g><rect class="bar elements ordinary" id="0" q0="73" o0="0" x="5" y="9" fill="#66a61e" width="20" height="231"></rect><rect class="bar elements ordinary" id="1" q0="73.34028252517285" o0="1" x="29" y="8" fill="#66a61e" width="20" height="232"></rect><rect class="bar elements ordinary" id="2" q0="72.9653315629157" o0="2" x="53" y="10" fill="#66a61e" width="20" height="230"></rect><rect class="bar elements ordinary" id="3" q0="74.31280681186632" o0="3" x="77" y="5" fill="#66a61e" width="20" height="235"></rect><rect class="bar elements ordinary" id="4" q0="74.41123672287168" o0="4" x="101" y="5" fill="#66a61e" width="20" height="235"></rect><rect class="bar elements ordinary" id="5" q0="75.17512708009241" o0="5" x="125" y="3" fill="#66a61e" width="20" height="237"></rect><rect class="bar elements ordinary" id="6" q0="75.18327200898251" o0="6" x="149" y="3" fill="#66a61e" width="20" height="237"></rect><rect class="bar elements ordinary" id="7" q0="74.55714906800908" o0="7" x="173" y="5" fill="#66a61e" width="20" height="235"></rect><rect class="bar elements ordinary" id="8" q0="75.2780276355189" o0="8" x="197" y="2" fill="#66a61e" width="20" height="238"></rect><rect class="bar elements ordinary" id="9" q0="75.32611951451123" o0="9" x="221" y="2" fill="#66a61e" width="20" height="238"></rect><rect class="bar elements ordinary" id="10" q0="75.29744749845595" o0="10" x="245" y="2" fill="#66a61e" width="20" height="238"></rect><rect class="bar elements ordinary" id="11" q0="76" o0="11" x="269" y="0" fill="#66a61e" width="20" height="240"></rect><rect class="bar elements ordinary" id="12" q0="62.40536478437488" o0="12" x="293" y="43" fill="#66a61e" width="20" height="197"></rect><rect class="bar elements ordinary" id="13" q0="55.6912383228965" o0="13" x="317" y="64" fill="#66a61e" width="20" height="176"></rect><rect class="bar elements ordinary" id="14" q0="47" o0="14" x="341" y="92" fill="#66a61e" width="20" height="148"></rect><g class="axis axis--x" transform="translate(0,240)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><path class="domain" stroke="currentColor" d="M0.5,6V0.5H366.1258621505212V6"></path><g class="tick" opacity="1" transform="translate(15.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord0</text></g><g class="tick" opacity="1" transform="translate(39.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord1</text></g><g class="tick" opacity="1" transform="translate(63.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord2</text></g><g class="tick" opacity="1" transform="translate(87.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord3</text></g><g class="tick" opacity="1" transform="translate(111.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord4</text></g><g class="tick" opacity="1" transform="translate(135.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord5</text></g><g class="tick" opacity="1" transform="translate(159.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord6</text></g><g class="tick" opacity="1" transform="translate(183.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord7</text></g><g class="tick" opacity="1" transform="translate(207.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord8</text></g><g class="tick" opacity="1" transform="translate(231.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord9</text></g><g class="tick" opacity="1" transform="translate(255.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord10</text></g><g class="tick" opacity="1" transform="translate(279.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord11</text></g><g class="tick" opacity="1" transform="translate(303.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord12</text></g><g class="tick" opacity="1" transform="translate(327.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord13</text></g><g class="tick" opacity="1" transform="translate(351.5,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">ord14</text></g></g><g class="axis axis--y" fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><path class="domain" stroke="currentColor" d="M-6,240.5H0.5V0.5H-6"></path><g class="tick" opacity="1" transform="translate(0,240.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">0</text></g><g class="tick" opacity="1" transform="translate(0,208.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">10</text></g><g class="tick" opacity="1" transform="translate(0,177.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">20</text></g><g class="tick" opacity="1" transform="translate(0,145.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">30</text></g><g class="tick" opacity="1" transform="translate(0,114.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">40</text></g><g class="tick" opacity="1" transform="translate(0,82.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">50</text></g><g class="tick" opacity="1" transform="translate(0,51.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">60</text></g><g class="tick" opacity="1" transform="translate(0,19.5)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">70</text></g><text transform="rotate(-90)" y="6" dy="0.71em" text-anchor="end"></text></g><text class="title" text-anchor="middle" font-size="10.799999999999999" x="182.8129310752606" y="-14.399999999999999">The Value</text></g></svg>'

    svg_number = 7
    need_text = True


    a_numpy, id_array, soup, text, focus_array = parse_svg_string(svg_string, min_element_num=svg_number, simple = True, need_text = need_text, need_focus = True)
    # verify_parsed_results()
    # parse_unknown_svg_visual_elements(svg_string)
        # print(svg_string)
    # a_numpy, id_array = parse_svg_string(svg_string)
    # print("numpy's size", a_numpy.shape)

    # # uniform_elements, data, soup = parse_unknown_svg(svg_string)
    # print(a_numpy)
