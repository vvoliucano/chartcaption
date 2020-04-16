import json
import svgwrite


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


def generate_visualization(data):
	names = data["names"]
	dates = data["dates"]
	content = data["content"]
	width = data["width"]
	height = data["height"]

	content_width = width
	content_height = height

	names_len = len(names)
	dates_len = len(dates)
	button_width = content_width / dates_len
	button_height = content_height / names_len

	svg = svgwrite.Drawing(filename = data["filename"], size=(width, height))

	for i in range(dates_len):
		for j in range(names_len):
			rect = svg.rect(insert=(button_width * i, button_height * j), size=(button_width, button_height), fill="red")
			svg.add(rect)

	svg.save()

	# print(data)



if __name__ == '__main__':
	line_number = 10
	col_number = 7
	data = [[(i + 1 + j) for i in range(col_number)] for j in range(line_number)]
	names = ["国家" + str(i) for i in range(line_number)]
	dates = ["4月" + str(i) + "日" for i in range(col_number)]
	data = {"names": names, "dates": dates, "content": data, "filename": "try.svg", "width": 100, "height": 140, "left_text": 0.2}
	generate_visualization(data)