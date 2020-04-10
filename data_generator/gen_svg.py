import svgwrite

def draw_barchart(bar_values, colors, canvas_width, canvas_height, border, padding, file_name):
    bar_num = len(bar_values)
    assert(bar_num != 0)
    svg = svgwrite.Drawing(filename=file_name, debug=True, size=(canvas_width, canvas_height))
    rect_g = svg.add(svg.g())
    border_width = canvas_width * border / 2
    bar_width = (canvas_width - border_width * 2) * (1-padding) / bar_num
    bar_padding = 0
    if bar_num != 1:
        bar_padding = canvas_width * padding / (bar_num-1)
    for i, bh in enumerate(bar_values):
        rect = svg.rect(insert=(border_width+i*bar_width+i*bar_padding, canvas_height-bh), size=(bar_width, bh), fill=colors[i])
        rect_g.add(rect)
    svg.save()
    return 

def draw_barchart_with_text(bar_values, colors, canvas_width, canvas_height, border, padding, file_name, x_axis_list):
    
    bar_num = len(bar_values)

    assert(bar_num != 0)
    # 新建SVG文件
    svg = svgwrite.Drawing(filename=file_name, debug=True, size=(canvas_width, canvas_height))
    # 添加一个组放置矩形
    rect_g = svg.add(svg.g())
    border_width = canvas_width * border / 2
    bar_width = (canvas_width - border_width * 2) * (1-padding) / bar_num
    bar_padding = 0
    if bar_num != 1:
        bar_padding = canvas_width * padding / (bar_num-1)
        # 组与组之间的间隙


    for i, bh in enumerate(bar_values):
        rect = svg.rect(insert=(border_width+i*bar_width+i*bar_padding, canvas_height-bh), size=(bar_width, bh), fill=colors[i])
        rect_g.add(rect)

    text_g = svg.add(svg.g())
    # classsvgwrite.text.Text(text, insert=None, x=None, y=None, dx=None, dy=None, rotate=None, **extra)
   
    for i, bh in enumerate(bar_values):
        text = svg.text(x_axis_list[i], insert=(border_width+i*bar_width+i*bar_padding, canvas_height))
        text_g.add(text)
    # print(svg)
    # print(file_name)

    svg.save()
    return 


if __name__ == '__main__':
    draw_barchart([20, 23, 40, 10], ["red", "steelblue", "yellow", "blue"], 100, 100, 0.1, 0.1, "test.svg")
