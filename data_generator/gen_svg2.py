import svgwrite

def draw_barchart(bar_values_all, colors_all, vsettings, color_set, svg_file_name):
    type1 = vsettings['type1']
    type2 = vsettings['type2']
    if type2 == 'grouped':
        draw_grouped_barchart(bar_values_all, colors_all, vsettings, color_set, svg_file_name)
    elif type2 == 'stacked':
        draw_stacked_barchart(bar_values_all, colors_all, vsettings, color_set, svg_file_name)
    else:
        raise(Exception(type1 + " " + type2 + ": not supported"))
    return

def draw_grouped_barchart(bar_values_all, colors_all, vsettings, color_set, svg_file_name):
    attr_num = len(bar_values_all)
    assert(attr_num != 0)
    bar_num = len(bar_values_all[0])
    assert(bar_num != 0)
    zip_bar_values_all = list(zip(*bar_values_all))
    canvas_width = vsettings['canvas_width']
    canvas_height = vsettings['canvas_height']
    canvas_height = min(canvas_height, max(map(lambda x: max(x), zip_bar_values_all))+30)
    border = vsettings['bar_border']
    padding = vsettings['bar_padding']
    svg = svgwrite.Drawing(filename=svg_file_name, debug=True, size=(canvas_width, canvas_height))
    rect_g = svg.add(svg.g())
    border_width = canvas_width * border / 2
    bar_width = (canvas_width - border_width * 2) * (1-padding) / bar_num
    bar_padding = 0
    if bar_num != 1:
        bar_padding = canvas_width * padding / (bar_num-1)
    each_bar_width = bar_width / attr_num
    for i, bhs in enumerate(zip_bar_values_all):
        for j, bh in enumerate(bhs):
            rect = svg.rect(insert=(border_width+i*bar_width+i*bar_padding+j*each_bar_width, canvas_height-bh), size=(each_bar_width, bh), fill=color_set[colors_all[j][i]])
            rect_g.add(rect)
    svg.save()
    return 

def draw_stacked_barchart(bar_values_all, colors_all, vsettings, color_set, svg_file_name):
    attr_num = len(bar_values_all)
    assert(attr_num != 0)
    bar_num = len(bar_values_all[0])
    assert(bar_num != 0)
    canvas_width = vsettings['canvas_width']
    canvas_height = vsettings['canvas_height']
    zip_bar_values_all = list(zip(*bar_values_all))
    canvas_height = min(canvas_height, max(map(lambda x: sum(x), zip_bar_values_all))+30)
    border = vsettings['bar_border']
    padding = vsettings['bar_padding']
    svg = svgwrite.Drawing(filename=svg_file_name, debug=True, size=(canvas_width, canvas_height))
    rect_g = svg.add(svg.g())
    border_width = canvas_width * border / 2
    bar_width = (canvas_width - border_width * 2) * (1-padding) / bar_num
    bar_padding = 0
    if bar_num != 1:
        bar_padding = canvas_width * padding / (bar_num-1)
    each_bar_width = bar_width
    for i, bhs in enumerate(zip_bar_values_all):
        acc_bh = 0
        for j, bh in enumerate(bhs):
            rect = svg.rect(insert=(border_width+i*bar_width+i*bar_padding, canvas_height-bh-acc_bh), size=(each_bar_width, bh), fill=color_set[colors_all[j][i]])
            acc_bh += bh
            rect_g.add(rect)
    svg.save()
    return 
if __name__ == '__main__':
    draw_barchart([20, 23, 40, 10], ["red", "steelblue", "yellow", "blue"], 100, 100, 0.1, 0.1, "test.svg")
