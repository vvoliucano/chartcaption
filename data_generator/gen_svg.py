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

if __name__ == '__main__':
    draw_barchart([20, 23, 40, 10], ["red", "steelblue", "yellow", "blue"], 100, 100, 0.1, 0.1, "test.svg")
