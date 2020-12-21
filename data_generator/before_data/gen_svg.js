#!/home/can.liu/nodejs/node-v12.16.2-linux-x64/bin/node

var d3 = require('d3');
var jsdom = require('jsdom');
//引入fs模块,直接用名字的方式引入
var fs = require('fs');

const {
  JSDOM
} = jsdom;

const {
  document
} = (new JSDOM('')).window;
global.document = document;



// svg.append("circle")
// 	.attr("cx",250)
// 	.attr("cy",250)
// 	.attr("r",250)
// 	.attr("fill","Red");

/*
    chart drawing
*/


let flag = true
let marginRate = .2
let fontRatio = .03 // fontRatio：字号基本值，是myheight (一个指定的固定值）对应的比例；比如legend的字号是1.5*fontRatio*myheight；我翻看了一下，legend axis title的字号都和fontRatio有点*k关系，但是当时代码写的很恶心，k比较随意不能统一改 
let legendHeightRatio = .06
let paddingValue = 0.5 // paddingValue是对于分组条形图的pad的比率
// let band_percentage = 0.5
// let myheight = document.getElementById('visualization').clientWidth * 0.95
// let mywidth = document.body.clientHeight * 0.8
// const myheight = flag? 400: 660
let myheight = 400


let round_value = 0
let tmp_value = Math.random()

if (tmp_value < 0.3)
  round_value = 1
else if (tmp_value > 0.7)
  round_value = 2



let aspect_ratio = 2
let mywidth = myheight * aspect_ratio



Array.prototype.unique = function() {
  var result = [],
    hash = {};
  for (var i = 0; i < this.length; i++) {
    if (!hash[this[i]]) {
      result.push(this[i]);
      hash[this[i]] = true;
    }
  }
  return result;
}

function deal_with_data(d) {
  // console.log(d.filename)
  let data_json = d
  if (d.hasOwnProperty("aspect_ratio")) {
    mywidth = myheight * d.aspect_ratio
    // console.log("hhh, aspect_ratio", d.aspect_ratio)
  }
  if (d.hasOwnProperty("paddingValue")) {
    paddingValue = d.paddingValue
  }

  switch (d['type']) {
    case 'ccq':
      deal_with_ccq(d)
      break;
    case 'ocq':
      deal_with_ocq(d)
      break;
    case 'cq':
      deal_with_cq(d)
      break;
    case 'oq':
      deal_with_oq(d)
      break;
    case 'qq':
      deal_with_qq(d)
      break;
    default:
      {
        console.log(d["type"])
        console.log('I can not handle this kind of data!')
      }

  }
  d3.select(document.body).select('svg').attr('height', 500)
  d3.select(document.body).select('svg').select('.main_canvas').attr('transform', 'translate(80,80)')
  // if (is_show) {
  //   let sent_data = { data_string: JSON.stringify(data_json),
  //                 svg_string: get_svg_string,
  //                 major_name: major_name,
  //                 second_name: second_name
  //               }
  //   get_machine_answer(sent_data)
  // }
}

function judge_type(data) {
  let d = data.data_array[0]
  let o0 = d.hasOwnProperty('o0')
  let o1 = d.hasOwnProperty('01')
  let c0 = d.hasOwnProperty('c0')
  let c1 = d.hasOwnProperty('c1')
  let q0 = d.hasOwnProperty('q0')
  let q1 = d.hasOwnProperty('q1')

  if (o0 && c0 && q0 && !o1 && !c1 && !q1) {
    return 'ocq'
  } else if (c0 && c1 && q0 && !o0 && !o1 && !q1) {
    return 'ccq'
  } else if (c0 && !c1 && q0 && !o0 && !o1 && !q1) {
    return 'cq'
  } else if (!c0 && !c1 && q0 && o0 && !o1 && !q1) {
    return 'oq'
  }
}

function deal_with_oq(data) {
  if (data.vis_type === "load_bar_chart_1d") {
    load_bar_chart_1d(data, 'o0', '', 'q0')
  } 
  else if (data.vis_type === "load_bar_chart_1d_horizontal") {
    load_bar_chart_1d_horizontal(data, 'o0', '', 'q0')
  }
  else if (data.vis_type === "load_line_chart_1d") {
    load_line_chart_1d(data, data['color'], "o0", "q0")
  }
  major_name = 'o0'
  second_name = ''
}

function deal_with_cq(data) {
  if (Math.random() > 0.5) {
    load_bar_chart_1d(data, 'c0', 'c1', 'q0')
  } else {
    load_bar_chart_1d_horizontal(data, 'c0', 'c1', 'q0')
  }
  major_name = 'c0'
  second_name = 'c1'
}

function deal_with_ccq(data) {
  if (Math.random() > 0.5) {
    load_group_bar_chart(data, 'c0', 'c1', 'q0')
  } else {
    load_stack_bar_chart_horizontal(data, 'c0', 'c1', 'q0')
  }
}


// 将决策权统一转移至数据

function deal_with_ocq(data) {
  if (data.vis_type === "load_stack_bar_chart_horizontal") {
    load_stack_bar_chart_horizontal(data, 'o0', 'c0', 'q0')
  } else if (data.vis_type === "load_stack_bar_chart") {
    load_stack_bar_chart(data, 'o0', 'c0', 'q0')
  } else if (data.vis_type === "load_group_bar_chart_horizontal") {
    load_group_bar_chart_horizontal(data, 'o0', 'c0', 'q0')
  } else if (data.vis_type === "load_group_bar_chart") {
    load_group_bar_chart(data, 'o0', 'c0', 'q0')
  } else if (data.vis_type === "load_line_chart") {
    // data = CCQ2CQQ(data)
    load_line_chart(data)
  }
}

function deal_with_qq(data) {
  load_scatter_plot(data, data["color"], 'q0', 'q1')
}

let extent = function(array, key) {
  let yMin = Infinity
  let yMax = -Infinity
  for (v of array) {
    yMin = Math.min(yMin, v[key])
    yMax = Math.max(yMax, v[key])
  }
  yMin = yMin < 0 ? yMin : 0
  return [yMin, yMax]
}

function load_line_chart(data){

  let marginRate = 0.1

  let height = myheight * (1 - 2 * marginRate)
  let width = mywidth * (1 - 2 * marginRate)
  // let windowWidth =  document.getElementById('visualization').clientWidth * 0.95
  // let windowHeight = document.body.clientHeight * 0.8
  let svg = d3.select(document.body).append("svg")
    .attr('id', 'mySvg')
    .attr('viewBox', '0 0 ' + String(mywidth) + ' ' + String(myheight))
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', mywidth)
    .attr('height', myheight)

  let canvas_g = svg.append('g')
    .attr('transform', 'translate(' + width * marginRate + ',' + (height * marginRate) + ')')
    .classed('main_canvas', true)

    // Add X axis --> it is a date format
  let xScale = d3.scaleLinear()
    .domain([0, data['o0'].length - 1])
    .range([0, width]);

  // console.log("其长度在于？", data['o0'].length)

  canvas_g.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(xScale).ticks(data["o0"].length).tickFormat(d=>data["o0"][d]))

  canvas_g.append('text')
    .text(data["title"])
    .attr("x", width/2)
    .attr("text-anchor", "middle")
    .attr('font-size', width / 20)


  // Add Y axis
  let yScale = d3.scaleLinear()
    .domain( [0, Math.max(...data["data_array"].map(d=>d.q0)) * 1.2])
    .range([height, 0]);

  canvas_g.append("g")
    .call(d3.axisLeft(yScale));

  let group_by_result = get_data_by_cat(data)

  // console.log(group_by_result)

  canvas_g.append('g')
    .attr("id", "content")
    .selectAll('path')
    .data(group_by_result)
    .enter()
    .append('path')
    .attr("d", d3.line()
        .x(function(d) { return xScale(+d.o0) })
        .y(function(d) { return yScale(+d.q0) })
    )
    .attr("stroke", (d, i) => data['color'][i])
    .style("stroke-width", 4)
    .style("fill", "none")


  // canvas_g.append('g')
  //   .attr('id', "legend")
  //   .attr('transform', "translate(" + width + ", 0)")
  //   .selectAll('')

  let canvasHeight = height
  let canvasWidth = width
  let legendHeight = canvasHeight * legendHeightRatio
  let fontSize = canvasHeight * fontRatio * 1.5
  let rectHeight = legendHeight * .9
  let rectWidth = canvasWidth * .03
  if (flag) rectWidth = rectHeight
  let cell = data["c0"].map((d, i) => ({
    'name': d,
    'color': data.color[i]
  }))
  // console.log(data['c0'])
  // console.log(cell)
  let canvas = canvas_g.append('g')
    .attr('transform', 'translate(' + String(canvasWidth) + ',0)')
    .attr('class', 'legend-wrap')

  let legends = canvas.selectAll('.legend')
    .data(cell)
    .enter()
    .append('g')
    .attr('transform', (d, i) => 'translate(0,' + String(i * legendHeight) + ')')

  // legends.append('rect')
  //   .attr('width', rectWidth)
  //   .attr('height', rectHeight)
  //   .attr('fill', d => d.color)
  //   .attr('id', (d, i) => 'color-' + String(i))
  //   .attr('color-data', d => d.color)
  //   .attr('custom-id', (d, i) => i)
  //   .attr('data-toggle', 'popover')
  //   .attr('data-container', 'body')
  //   .attr('data-placement', 'right')
  //   .attr('onclick', 'addColorPicker(this)')

  legends.append('line')
    .attr('x1', 0)
    .attr('x2', rectWidth)
    .attr('y1', rectHeight / 2)
    .attr('y2', rectHeight / 2)
    .attr('stroke', d => d.color)
    .attr('id', (d, i) => 'color-' + String(i))
    .attr('stroke-width', 4)

  legends.append('text')
    .attr('x', rectWidth + .2 * fontSize)
    .attr('y', fontSize)
    .attr('text-anchor', 'start')
    .attr('font-size', fontSize)
    .text(d => d.name)

    
  function get_data_by_cat(data){
    let result = new Array()
    let cat_num = data['c0'].length
    for (let i = 0; i < cat_num; i ++ )
    {
      result[i] = new Array()
    }
    let item_num = data['data_array'].length
    for (let i = 0; i < item_num; i ++)
    {
      datum = data.data_array[i]
      result[datum['c0']].push(datum)
    }
    // console.log('result: ', result)
    return result
  }

}


function load_line_chart_1d(data, cat_color, x_axis, y_axis){

  let marginRate = 0.1

  let height = myheight * (1 - 2 * marginRate)
  let width = mywidth * (1 - 2 * marginRate)
  // let windowWidth =  document.getElementById('visualization').clientWidth * 0.95
  // let windowHeight = document.body.clientHeight * 0.8
  let svg = d3.select(document.body).append("svg")
    .attr('id', 'mySvg')
    .attr('viewBox', '0 0 ' + String(mywidth) + ' ' + String(myheight))
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', mywidth)
    .attr('height', myheight)

  let canvas_g = svg.append('g')
    .attr('transform', 'translate(' + width * marginRate + ',' + (height * marginRate) + ')')
    // .classed('main_canvas', true)

    // Add X axis --> it is a date format
  let xScale = d3.scaleLinear()
    .domain([0, data['o0'].length - 1])
    .range([0, width]);

  // console.log("其长度在于？", data['o0'].length)

  canvas_g.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(xScale).ticks(data["o0"].length).tickFormat(d=>data["o0"][d]))

  canvas_g.append('text')
    .text(data["title"])
    .attr("x", width/2)
    .attr("text-anchor", "middle")
    .attr('font-size', width / 20)


  // Add Y axis
  let yScale = d3.scaleLinear()
    .domain( [0, Math.max(...data["data_array"].map(d=>d.q0)) * 1.2])
    .range([height, 0]);

  canvas_g.append("g")
    .call(d3.axisLeft(yScale));

  canvas_g.append('path')
    .datum(data.data_array)
    .attr("d", d3.line()
      .x(function(d) { return xScale(+d.o0) })
      .y(function(d) { return yScale(+d.q0) })
    )
    .attr("stroke", cat_color[0])
    .style("stroke-width", 4)
    .style("fill", "none")

}



function CQQ(data, cat_color, cat_x, cat_y, position = 'vertical', tag = 'scatter') {
  this.tag = tag
  // initial chart set up
  this.height = myheight
  this.width = mywidth
  // let windowWidth =  document.getElementById('visualization').clientWidth * 0.95
  // let windowHeight = document.body.clientHeight * 0.8
  this.svg = d3.select(document.body).append("svg")
    .attr('id', 'mySvg')
    .attr('viewBox', '0 0 ' + String(this.width) + ' ' + String(this.height))
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', this.width)
    .attr('height', this.height)

  this.g = this.svg.append('g')
    .attr('transform', 'translate(' + this.width * marginRate + ',' + this.height * marginRate + 100 + ')')
    .classed('main_canvas', true)
  if (flag) {
    this.g.attr('transform', 'translate(' + this.width * marginRate + ',' + this.height * 0.1 + ')')
  }
  this.g.append('g')
    .attr('class', 'brush')
  // database set up
  this.position = position
  this.data = data
  this.majorName = cat_x
  this.secondName = cat_y
  this.category = cat_color
  // scale set up
  this.scaleHeight = [this.height * (1 - 2 * marginRate), 0]
  this.scaleWidth = [0, this.width * (1 - 2 * marginRate)]
  this.xScale = d3.scaleLinear()
  this.yScale = d3.scaleLinear()
  if (this.position === 'horizontal') {
    this.xScale.rangeRound(this.scaleHeight)
    this.yScale.rangeRound(this.scaleWidth)
    if (this.tag != 'scatter') {
      console.log('HELP!!!, the tag here is not scatter')
      this.xScale = d3.scalePoint()
        .domain(this.data['o0'])
        .range(this.scaleHeight)
    }
  } else {
    this.xScale.rangeRound(this.scaleWidth)
    this.yScale.rangeRound(this.scaleHeight)
    if (this.tag != 'scatter') {
      this.xScale = d3.scalePoint()
        .domain(this.data['o0'])
        .range(this.scaleWidth)
    }
  }
  // for (let i = 0; i < this.data['o0'].length; i ++){
  //   console.log(i, "...", this.xScale(i))
  // }
  // console.log(this.xScale(0))

  // console.log('WHERE', this.xScale(2013), this.xScale(2012))
}

CQ.prototype.drawTitle = function(title) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let fontSize = canvasWidth /20

  this.g.append('text')
    .attr('class', 'title')
    .attr('text-anchor', 'middle')
    .attr('font-size', fontSize)
    .attr('x', canvasWidth / 2)
    .attr('y', -3 * fontSize)
    .text(title)

  return this
}

CQQ.prototype.drawAxis = function() {

  let yMin = Infinity
  let yMax = -Infinity
  let xMin = Infinity
  let xMax = -Infinity
  for (v of this.data.data_array) {
    yMax = Math.max(yMax, v[this.secondName])
    xMax = Math.max(xMax, v[this.majorName])
    yMin = Math.min(yMin, v[this.secondName])
    xMin = Math.min(xMin, v[this.majorName])
  }
  // WDT改了一下这里，避免Scatterplot某些点跑到轴上
  yMax += 1
  xMax += 0.1
  yMin = 0
  xMin -= 0.1
  this.yScale.domain([yMin, yMax])
  if (this.tag == 'scatter') {
    this.xScale.domain([xMin, xMax])
  }
  if (this.position === 'horizontal') {
    let axisLeft = d3.axisLeft(this.xScale)
    if (flag) {
      let canvasWidth = this.width * (1 - 2 * marginRate - 0.1)
      axisLeft = d3.axisLeft(this.xScale).ticks(5).tickSize(-canvasWidth)
    }
    let yAxis = this.g.append('g')
      .attr('class', 'axis')
      .call(axisLeft)
    let unit1 = yAxis.append('text')
      .attr('y', 6)
      .attr('dy', '0.71em')
      .attr('text-anchor', 'end')
      .attr('font-size', '20px')
      .text(this.data.unit1)


    let xAxis = this.g.append('g')
      .attr('class', 'axis')
      .attr('transform', 'translate(0,' + this.height * (1 - 2 * marginRate) + ')')
      .call(d3.axisBottom(this.yScale))

    let unit2 = xAxis.append('text')
      .attr('transform', 'translate(' + String(this.scaleWidth[1]) + ',' + this.scaleHeight[0] + ')')
      .attr('text-anchor', 'end')
      .attr('class', 'text-truncate')
      .attr('font-size', '20px')
      .text(this.data.unit2)

    if (flag) {
      let axis = d3.select(document.body).selectAll('.axis')
      axis.selectAll('.tick').selectAll('line').remove()
      axis.selectAll('.domain').remove()
      axis.selectAll('text').style('font-size', '20').style('font-family', 'Oxygen').style('fill', '#253039')
      xAxis.call(g => g.selectAll('.tick:not(:first-of-type) line')
        .style('stroke-opacity', 0.5))
      // .style('stroke-dasharray', '2,2'))
    }
    return this
  }
  let axisLeft = d3.axisLeft(this.yScale)
  if (flag) {
    let canvasWidth = this.width * (1 - 2 * marginRate)
    axisLeft = d3.axisLeft(this.yScale).ticks(5).tickSize(-canvasWidth)
  }
  let xAxis = this.g.append('g')
    .classed('axis', true)
    .attr('transform', 'translate(0,' + this.height * (1 - 2 * marginRate) + ')')
    .call(d3.axisBottom(this.xScale).ticks(3))
  let unit1 = this.g.append('text')
    .attr('transform', `translate(${this.scaleWidth[1]/2}, ${this.height - 10})`)
    .attr('text-anchor', 'middle')
    .attr('font-size', '20px')
    .text(this.data.unit1)

  let yAxis = this.g.append('g')
    .classed('axis', true)
    .call(axisLeft)
  let unit2 = this.g.append('text')
    .attr('transform', `translate(${-35}, ${this.height/2}) rotate(-90)`)
    .attr('text-anchor', 'start')
    .attr('font-size', '20px')
    .text(this.data.unit2)
  if (flag) {
    let axis = d3.select(document.body).selectAll('.axis')
    axis.selectAll('.domain').remove()
    xAxis.selectAll('line').remove()
    axis.selectAll('text').style('font-size', '20').style('font-family', 'Oxygen').style('fill', '#253039')
    yAxis.selectAll('.tick:not(:first-of-type) line')
      .style('stroke-opacity', 0.3)
    // .style('stroke-dasharray', '1,1')
  }
  return this
}

CQQ.prototype.drawScatterPlot = function() {
  // console.log(this)

  if (this.position === 'horizontal') {
    this.g.selectAll('.circle')
      .data(this.data.data_array)
      .enter().append('circle')
      .attr('class', function(d) {
        return 'element_' + String(d['id'])
      })
      .classed('circle', true)
      .classed('elements', true)
      .attr('id', d => d['id'])
      .attr(this.secondName, d => d[this.secondName])
      .attr(this.majorName, d => d[this.majorName]) //.y(d => this.xScale(this.data[this.majorName][d[this.majorName]]))
      .attr('cx', d => this.xScale(d[this.secondName]))
      .attr('cy', d => this.yScale(this.data[this.majorName][d[this.majorName]]))
      .attr('r', '0.5vh')
      .attr('alpha', 0.8)
      .attr('fill', d => this.data.color[d[this.category]])
      .classed('ordinary', true)
  }

      


  this.g.selectAll('.circle')
    .data(this.data.data_array)
    .enter().append('circle')
    .attr('class', d => 'element_' + String(d['id']))
    .classed('circle', true)
    .classed('elements', true)
    .attr('id', d => d['id'])
    .attr(this.secondName, d => d[this.secondName])
    .attr(this.majorName, d => d[this.majorName])
    .attr('cx', d => this.xScale(this.data[this.majorName][d[this.majorName]]))
    .attr('cy', d => this.yScale(d[this.secondName]))
    .attr('r', '0.5vh')
    .attr('alpha', 0.8)
    .attr('fill', d => this.data.color[d[this.category]])
    .classed('ordinary', true)
}

CQQ.prototype.countScatter = function() {
  let k = this.data.data_array[0][this.category]
  for (v of this.data.data_array) {
    if (v[this.category] != k) {
      return true
    }
  }
  return false
}

CQQ.prototype.drawLegend = function(cat_color) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let legendHeight = canvasHeight * legendHeightRatio
  let fontSize = canvasHeight * fontRatio * 1.5
  let rectHeight = legendHeight * .9
  let rectWidth = canvasWidth * .03
  if (flag) rectWidth = rectHeight
  let cell = this.data[this.category].map((d, i) => ({
    'name': d,
    'color': this.data.color[i]
  }))
  let canvas = this.g.append('g')
    .attr('transform', 'translate(' + String(canvasWidth) + ',0)')
    .attr('class', 'legend-wrap')
  let legends = canvas.selectAll('.legend')
    .data(cell)
    .enter()
    .append('g')
    .attr('transform', (d, i) => 'translate(0,' + String(i * legendHeight) + ')')
  legends.append('rect')
    .attr('width', rectWidth)
    .attr('height', rectHeight)
    .attr('fill', d => d.color)
    .attr('id', (d, i) => 'color-' + String(i))
    .attr('color-data', d => d.color)
    .attr('custom-id', (d, i) => i)
    .attr('data-toggle', 'popover')
    .attr('data-container', 'body')
    .attr('data-placement', 'right')
    .attr('onclick', 'addColorPicker(this)')
  legends.append('text')
    .attr('x', rectWidth + .2 * fontSize)
    .attr('y', fontSize)
    .attr('text-anchor', 'start')
    .attr('font-size', fontSize)
    .text(d => d.name)
  return canvas
}

CQQ.prototype.drawTitle = function(title) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let fontSize = canvasHeight * fontRatio

  this.g.append('text')
      .attr('class', 'title')
      .attr('text-anchor', 'middle')
      .attr('font-size', fontSize * 1.5)
      .attr('x', canvasWidth / 2)
      .attr('y', -1 * fontSize)
      .text(title)

  return this
}

CQQ.prototype.drawLine = function() {
  // console.log(this.data, this.xScale(2013))
  let line
  // console.log("this.secondName", this.secondName)
  // console.log("this.majorName", this.majorName)

  if (this.position == 'horizontal') {
    line = d3.line()
      .x(d => this.yScale(d[this.secondName]))
      .y(d => this.xScale(this.data[this.majorName][d[this.majorName]]))
      .curve(d3.curveMonotoneX)
  } else {
    line = d3.line()
      .x(d => this.xScale(this.data[this.majorName][d[this.majorName]]))
      .y(d => this.yScale(d[this.secondName]))
      .curve(d3.curveMonotoneX)
  }
  for (i in this.data[this.category]) {
    let data = []
    for (item of this.data.data_array) {
      if (item[this.category] == i) {
        data.push(item)
      }
    }
    this.g.append('path')
      .datum(data)
      .attr('class', 'line')
      .attr('d', line)
      .attr('stroke', this.data.color[i])
      .style('fill', 'none')
  }
}

function CQ(data, cat_position, cat_color, quantity, position = 'vertical') {
  // initial chart set up
  this.svg = d3.select(document.body).append('svg').attr('id', 'mySvg')
  this.height = myheight
  this.width = mywidth
  this.svg.attr('viewBox', '0 0 ' + String(this.width) + ' ' + String(this.height))
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('height', myheight)
    .attr('width', mywidth)
  this.g = this.svg.append('g')
    .attr('transform', 'translate(' + this.width * marginRate + ',' + this.height * marginRate + ')')
    .classed('main_canvas', true)
  this.g.append('g')
    .attr('class', 'brush')
  // database set up
  this.position = position
  this.data = data
  this.majorName = cat_position
  this.secondName = cat_color
  this.quantity = quantity
  // scale set up
  this.scaleHeight = [this.height * (1 - 2 * marginRate), 0]
  this.scaleWidth = [0, this.width * (1 - 2 * marginRate)]
  this.xScale = d3.scaleBand()
    .padding(paddingValue)
    .domain(this.data[this.majorName])
  this.yScale = d3.scaleLinear()
  if (this.position === 'horizontal') {
    this.xScale.rangeRound(this.scaleHeight)
    this.yScale.rangeRound(this.scaleWidth)
  } else {
    this.xScale.rangeRound(this.scaleWidth)
    this.yScale.rangeRound(this.scaleHeight)
  }
}

CQ.prototype.drawBarChart = function() {
  if (this.position === 'horizontal') {
    this.drawBarChart_Horizontal()
    return
  }
  this.xScale.domain(this.data[this.majorName])
  let yMin = Infinity
  let yMax = -Infinity
  for (v of this.data.data_array) {
    yMin = 0
    yMax = Math.max(yMax, v[this.quantity])
  }
  this.yScale.domain([yMin, yMax])

  this.g.selectAll('.bar')
    .data(this.data.data_array)
    .enter().append('rect')
    .attr('class', 'bar')
    .classed('elements', true)
    .attr('id', d => d['id'])
    .attr(this.quantity, d => d[this.quantity])
    .attr(this.majorName, d => d[this.majorName])
    .attr('x', d => this.xScale(this.data[this.majorName][d[this.majorName]]))
    .attr('y', d => this.yScale(d[this.quantity]))
    .attr('fill', (d, i) => this.data.color[0]) // 普通柱形图 按顺序赋色
    .classed('ordinary', true)
    .attr('width', this.xScale.bandwidth())
    .attr('height', d => this.scaleHeight[0] - this.yScale(d[this.quantity]))

  this.g.append('g')
    .attr('class', 'axis axis--x')
    .attr('transform', 'translate(0,' + this.scaleHeight[0] + ')')
    .call(d3.axisBottom(this.xScale))

  this.g.append('g')
    .attr('class', 'axis axis--y')
    .call(d3.axisLeft(this.yScale))
    .append('text')
    .attr('transform', 'rotate(-90)')
    .attr('y', 6)
    .attr('dy', '0.71em')
    .attr('text-anchor', 'end')
    .text(this.data.unit)
}

CQ.prototype.drawBarChart_Horizontal = function() {
  this.xScale.domain(this.data[this.majorName])
  let yMin = Infinity
  let yMax = -Infinity
  for (v of this.data.data_array) {
    yMin = 0
    yMax = Math.max(yMax, v[this.quantity])
  }
  this.yScale.domain([yMin, yMax])
  this.g.selectAll('.bar')
    .data(this.data.data_array)
    .enter().append('rect')
    .attr('class', 'bar')
    .classed('elements', true)
    .classed('ordinary', true)
    .attr('y', d => this.xScale(this.data[this.majorName][d[this.majorName]]))
    .attr('x', d => 0)
    .attr('id', d => d['id'])
    .attr(this.quantity, d => d[this.quantity])
    .attr(this.majorName, d => d[this.majorName])
    .attr('fill', (d, i) => this.data.color[0])
    .attr('height', this.xScale.bandwidth())
    .attr('width', d => this.yScale(d[this.quantity]))

  this.g.append('g')
    .attr('class', 'axis axis--x')
    .call(d3.axisLeft(this.xScale))

  this.g.append('g')
    .attr('class', 'axis axis--y')
    // .attr('transform', 'translate(0,' + this.scaleHeight[0] + ')')

    .call(d3.axisTop(this.yScale))
  this.g.append('text')
    .attr('transform', 'translate(' + String(this.scaleWidth[1]) + ',0)')
    .attr('dy', '-1.2rem')
    .attr('text-anchor', 'end')
    .attr('font-size', this.height * (1 - 2 * marginRate) * fontRatio)
    .text(this.data.unit);
}

CQ.prototype.drawTitle = function(title) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let fontSize = canvasHeight * fontRatio
  this.g.append('text')
    .attr('class', 'title')
    .attr('text-anchor', 'middle')
    .attr('font-size', fontSize * 1.5)
    .attr('x', canvasWidth / 2)
    .attr('y', -2 * fontSize)
    .text(title)
  return this
}

CQ.prototype.drawPieChart = function() {
  this.g.attr('transform', 'translate(' + this.width / 2 + ',' + this.height / 2 + ')');
  let radius = Math.min(this.width, this.height) / 3
  let pie = d3.pie()
    .sort(null)
    .value(function(d) {
      return d['c0']
    })

  let path = d3.arc()
    .outerRadius(radius - 10)
    .innerRadius(0)

  let label = d3.arc()
    .outerRadius(radius - 40)
    .innerRadius(radius - 40)

  let arc = this.g.selectAll('.arc')
    .data(pie(this.data.data_array))
    .enter()
    .append('g')
    .attr('class', 'arc');

  arc.append('path')
    .attr('d', path)
    .attr('fill', (d, i) => this.data['color'][i]);

  arc.append('text')
    .attr('transform', function(d) {
      return 'translate(' + label.centroid(d) + ')';
    })
    .attr('dy', '0.35em')
    .text((d, i) => this.data['c0'][i]);

}

CQ.prototype.drawLegend = function(cat_color) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let legendHeight = canvasHeight * legendHeightRatio
  let fontSize = canvasHeight * fontRatio
  let rectHeight = legendHeight * .9
  let rectWidth = canvasWidth * .03
  let cell = this.data[cat_color].map((d, i) => ({
    'name': d,
    'color': this.data.color[i]
  }))
  let canvas = this.g.append('g')
    .attr('transform', 'translate(' + canvasWidth / 2 + ',' + -canvasHeight / 3 + ')')
    .attr('class', 'legend-wrap')
  let legends = canvas.selectAll('.legend')
    .data(cell)
    .enter()
    .append('g')
    .attr('transform', (d, i) => 'translate(0,' + i * legendHeight + ')')
  legends.append('rect')
    .attr('width', rectWidth)
    .attr('height', rectHeight)
    .attr('fill', d => d.color)
    .attr('id', (d, i) => 'color-' + String(i))
    .attr('color-data', d => d.color)
    .attr('custom-id', (d, i) => i)
    .attr('data-toggle', 'popover')
    .attr('data-container', 'body')
    .attr('data-placement', 'right')
    .attr('onclick', 'addColorPicker(this)')
  legends.append('text')
    .attr('x', rectWidth + .2 * fontSize)
    .attr('y', fontSize)
    .attr('text-anchor', 'start')
    .attr('font-size', fontSize)
    .text(d => d.name)
  return canvas
}

function CCQ(data, cat_position, cat_color, quantity, position = 'vertical') {
  // initial chart set up

  // let marginRate = 0.1 

  this.height = myheight
  this.width = mywidth
  // let windowWidth =  document.getElementById('visualization').clientWidth * 0.95
  // let windowHeight = document.body.clientHeight * 0.8
  this.svg = d3.select(document.body).append("svg")
    .attr('id', 'mySvg')
    // .attr('viewBox', '0 0 ' + String(this.width) + ' ' + String(this.height))
    // .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', this.width)
    .attr('height', this.height)
  this.g = this.svg.append('g')
    .attr('transform', 'translate(' + this.width * marginRate + ',' + this.height * marginRate + ')')
    .classed('main_canvas', true)
  if (flag) {
    this.g.attr('transform', 'translate(' + this.width * marginRate + ',' + this.height * 0.1 + ')')
  }
  // database set up
  this.position = position
  this.data = data
  this.majorName = cat_position
  this.secondName = cat_color
  this.quantity = quantity
  // scale set up
  this.scaleHeight = [this.height * (1 - 2 * marginRate), 0]
  this.scaleWidth = [0, this.width * (1 - 2 * marginRate)]
  this.xScale = d3.scaleBand()
    .padding(paddingValue)
    .domain(this.data[this.majorName])
  this.yScale = d3.scaleLinear()
  if (this.position === 'horizontal') {
    this.xScale.rangeRound(this.scaleHeight)
    this.yScale.rangeRound(this.scaleWidth)
  } else {
    this.xScale.rangeRound(this.scaleWidth)
    this.yScale.rangeRound(this.scaleHeight)
  }
}

CCQ.prototype.drawAxis = function() {
  let yMin = Infinity
  let yMax = -Infinity
  for (v of this.data.data_array) {
    yMin = 0
    yMax = Math.max(yMax, v[this.quantity])
  }
  this.yScale.domain([yMin, yMax])
  if (this.position === 'horizontal') {
    let axisBottom = d3.axisBottom(this.yScale)
    if (flag) {
      let canvasHeight = this.height * (1 - 2 * marginRate)
      axisBottom = d3.axisBottom(this.yScale).ticks(6).tickSize(-canvasHeight)
    }
    let yAxis = this.g.append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(this.xScale))

    let xAxis = this.g.append('g')
      .attr('class', 'axis')
      .attr('transform', 'translate(0,' + this.height * (1 - 2 * marginRate) + ')')
      .call(axisBottom)

    let unit = this.g.append('text')
      .attr('transform', `translate(${this.scaleWidth[1]/2}, ${this.height - 10})`)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-truncate')
      .attr('font-size', '20px')
      .text(this.data.unit)
    if (flag) {
      let axis = d3.select(document.body).selectAll('.axis')
      yAxis.selectAll('.tick').selectAll('line').remove()
      axis.selectAll('.domain').remove()
      axis.selectAll('text').style('font-size', '20').style('font-family', 'Oxygen').style('fill', '#253039')
      xAxis.call(g => g.selectAll('.tick:not(:first-of-type) line')
        .style('stroke-opacity', 0.5))
      // .style('stroke-dasharray', '2,2'))
    }
    return this
  }
  let xAxis = this.g.append('g')
    .attr('class', 'axis')
    .attr('transform', 'translate(0,' + this.height * (1 - 2 * marginRate) + ')')
    .call(d3.axisBottom(this.xScale).ticks(5))
  let axisLeft = d3.axisLeft(this.yScale)
  if (flag) {
    let canvasWidth = this.width * (1 - 2 * marginRate)
    axisLeft = d3.axisLeft(this.yScale).ticks(5).tickSize(-canvasWidth)
  }
  let yAxis = this.g.append('g')
    .attr('class', 'axis')
    .call(axisLeft)
  let unit = this.g.append('text')
    .attr('transform', `translate(${-35}, ${this.height/2}) rotate(-90)`)
    .attr('text-anchor', 'start')
    .attr('font-size', '20px')
    .text(this.data.unit)
  if (flag) {
    let axis = d3.select(document.body).selectAll('.axis')
    axis.selectAll('.domain').remove()
    xAxis.selectAll('line').remove()
    axis.selectAll('text').style('font-size', '20').style('font-family', 'Oxygen').style('fill', '#253039')
    yAxis.selectAll('.tick:not(:first-of-type) line')
      .style('stroke-opacity', 0.3)
    // .style('stroke-dasharray', '1,1')
  }
  return this
}

CCQ.prototype.drawStackBarChart = function(key) {
  // pre-processing for rect position
  let data = this.data[this.majorName].map(v => {
    return {
      [this.majorName]: v
    }
  })
  // let index = this.data[this.majorName].map(v => {return {}} )
  let index = {}
  let value = {}
  for (let majorName of this.data[this.majorName]) {
    index[majorName] = {}
    value[majorName] = {}
    for (let secondName of this.data[this.secondName]) {
      index[majorName][secondName] = null
      value[majorName][secondName] = null
    }
  }
  for (let d of this.data.data_array) {
    let majorIdx = d[this.majorName]
    let secondIdx = d[this.secondName]
    let majorName = this.data[this.majorName][majorIdx]
    let secondName = this.data[this.secondName][secondIdx]
    data[majorIdx][secondName] = d[this.quantity]
    index[majorName][secondName] = d.id
    value[majorName][secondName] = d[this.quantity]
  }
  let stack = d3.stack().keys(this.data[this.secondName]).order(d3.stackOrderNone).offset(d3.stackOffsetNone)
  let series = stack(data)
  let right = []
  let i = 0
  for (let second of series) {
    right.push([])
    for (let first of second) {
      right[i].push({
        'array': first,
        [this.majorName]: first.data[this.majorName],
        [this.secondName]: second.key
      })
    }
    ++i
  }
  // scale domain
  let total = data.map(v => {
    let sum = 0
    this.data[this.secondName]
      .forEach(name => {
        sum += v[name]
      })
    return sum
  })
  let maximum = d3.max(total)

  this.yScale.domain([0, maximum])

  // draw stack bar chart
  if (this.position === 'horizontal') {
    this.g.append('g')
      .selectAll('g')
      .data(right)
      .enter()
      .append('g')
      .attr('fill', (d, i) => this.data.color[i])
      .selectAll('rect')
      .data(d => d)
      .enter()
      .append('rect')
      .classed('bar', true)
      .classed('elements', true)
      .classed('ordinary', true)
      .attr('id', d => String(index[d[this.majorName]][d[this.secondName]]))
      .attr('class', d => 'element_' + String(index[d[this.majorName]][d[this.secondName]]))
      .attr(this.majorName, d => d[this.majorName])
      .attr(this.secondName, d => d[this.secondName])
      .attr(this.quantity, d => value[d[this.majorName]][d[this.secondName]])
      .attr('y', (d, i) => this.xScale(d[this.majorName]))
      .attr('x', d => this.yScale(d.array[0]))
      .attr('width', d => this.yScale(d.array[1]) - this.yScale(d.array[0]))
      .attr('height', this.xScale.bandwidth())
    return this
  } else {
    this.g.append('g')
      .selectAll('g')
      .data(right)
      .enter()
      .append('g')
      .attr('fill', (d, i) => this.data.color[i])
      .selectAll('rect')
      .data(d => d)
      .enter()
      .append('rect')
      .classed('bar', true)
      .classed('elements', true)
      .classed('ordinary', true)
      .attr('id', d => {
        return String(index[d[this.majorName]][d[this.secondName]])
      })
      .attr('class', d => 'element_' + String(index[d[this.majorName]][d[this.secondName]]))
      .attr(this.majorName, d => d[this.majorName])
      .attr(this.secondName, d => d[this.secondName])
      .attr(this.quantity, d => value[d[this.majorName]][d[this.secondName]])
      .attr('x', (d, i) => this.xScale(d[this.majorName]))
      .attr('y', d => this.yScale(d.array[1]))
      .attr('height', d => this.yScale(d.array[0]) - this.yScale(d.array[1]))
      .attr('width', this.xScale.bandwidth())
    return this
  }

}

CCQ.prototype.drawGroupBarChart = function() {
  let innerScale = d3.scaleBand().domain(this.data[this.secondName]).rangeRound([0, this.xScale.bandwidth()])
  this.yScale.domain(extent(this.data.data_array, this.quantity))
  if (this.position === 'horizontal') {
    let rects = this.g.append('g').attr('class', 'bars').selectAll('rect')
      .data(this.data.data_array)
      .enter()
      .append('rect')
      .attr('fill', (d, i) => this.data.color[d[this.secondName]])
      .attr('id', d => d.id)
      .attr('class', d => 'element_' + String(d.id))
      .classed('ordinary', true)
      .attr('x', 0)
      .attr('y', d => this.xScale(this.data[this.majorName][d[this.majorName]]) + innerScale(this.data[this.secondName][d[this.secondName]]) + 2)
      .attr('width', d => this.yScale(d[this.quantity]))
      .attr('height', innerScale.bandwidth() * 0.5)
    return this
  }

  let rects = this.g.append('g').attr('class', 'bars').selectAll('rect')
    .data(this.data.data_array)
    .enter()
    .append('rect')
    .attr('fill', (d, i) => this.data.color[d[this.secondName]])
    .attr('id', d => d.id)
    .attr('class', d => 'element_' + String(d.id))
    .classed('elements', true)
    .classed('ordinary', true)
    .attr('x', d => innerScale(this.data[this.secondName][d[this.secondName]]) + this.xScale(this.data[this.majorName][d[this.majorName]]) + 2)
    .attr('y', d => this.yScale(d[this.quantity]))
    .attr('height', d => this.scaleHeight[0] - this.yScale(d[this.quantity]))
    .attr('width', innerScale.bandwidth() * 0.9)

  rects.attr('rx', round_value)
    .attr('ry', round_value)
  return this
}

CCQ.prototype.drawTitle = function(title) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let fontSize = canvasHeight * fontRatio
  if (flag) {
    this.g.append('text')
      .attr('class', 'title')
      .attr('text-anchor', 'middle')
      .attr('font-size', fontSize * 4)
      .attr('x', canvasWidth / 2)
      .attr('y', -2 * fontSize - 0.1 * canvasHeight)
      .text(title.toUpperCase())
      .style('font-family', 'Oxygen')
      .style('font-weight', 'bold')
      .style('fill', '#253039')

  } else {
    this.g.append('text')
      .attr('class', 'title')
      .attr('text-anchor', 'middle')
      .attr('font-size', fontSize * 1.5)
      .attr('x', canvasWidth / 2)
      .attr('y', -2 * fontSize)
      .text(title)
  }
  return this
}

CCQ.prototype.drawLegend = function(cat_color) {
  let canvasHeight = this.height * (1 - 2 * marginRate)
  let canvasWidth = this.width * (1 - 2 * marginRate)
  let legendHeight = canvasHeight * legendHeightRatio
  let fontSize = canvasHeight * fontRatio * 1.5
  let rectHeight = legendHeight * .9
  let rectWidth = canvasWidth * .03
  if (flag) rectWidth = rectHeight
  let cell = this.data[cat_color].map((d, i) => ({
    'name': d,
    'color': this.data.color[i]
  }))
  let canvas = this.g.append('g')
    .attr('transform', 'translate(' + canvasWidth + ',0)')
    .attr('class', 'legend-wrap')
  let legends = canvas.selectAll('.legend')
    .data(cell)
    .enter()
    .append('g')
    .attr('transform', (d, i) => 'translate(0,' + i * legendHeight + ')')
  legends.append('rect')
    .attr('width', rectWidth)
    .attr('height', rectHeight)
    .attr('fill', d => d.color)
    .attr('id', (d, i) => 'color-' + String(i))
    .attr('color-data', d => d.color)
    .attr('custom-id', (d, i) => i)
    .attr('data-toggle', 'popover')
    .attr('data-container', 'body')
    .attr('data-placement', 'right')
    .attr('onclick', 'addColorPicker(this)')
  legends.append('text')
    .attr('x', rectWidth + .2 * fontSize)
    .attr('y', fontSize)
    .attr('text-anchor', 'start')
    .attr('font-size', fontSize)
    .text(d => d.name)
  return canvas
}

function load_stack_bar_chart(data, cat_position, cat_color, quantity) {
  chart = new CCQ(data, cat_position, cat_color, quantity)
  chart.drawAxis()
  chart.drawStackBarChart(cat_color)
  chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_stack_bar_chart_horizontal(data, cat_position, cat_color, quantity) {
  chart = new CCQ(data, cat_position, cat_color, quantity, 'horizontal')
  chart.drawAxis()
  chart.drawStackBarChart(cat_color)
  chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_bar_chart_1d(data, cat_position, cat_color, quantity) {
  let chart = new CQ(data, cat_position, cat_color, quantity)
  chart.drawBarChart()
  chart.drawTitle(data.title)
}

function load_bar_chart_1d_horizontal(data, cat_position, cat_color, quantity) {
  let chart = new CQ(data, cat_position, cat_color, quantity, 'horizontal')
  chart.drawBarChart()
  chart.drawTitle(data.title)
}

function load_line_plot(data, cat_color, x, y) {
  let chart = new CQQ(data, cat_color, x, y)
  chart.drawAxis()
  chart.drawLine()
  if (chart.countScatter())
    chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_line_plot_horizontal(data, cat_color, x, y) {
  let chart = new CQQ(data, cat_color, x, y, 'horizontal')
  chart.drawAxis()
  chart.drawLine()
  if (chart.countScatter())
    chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}


function load_group_bar_chart(data, cat_position, cat_color, quantity) {
  let chart = new CCQ(data, cat_position, cat_color, quantity)
  chart.drawAxis()
  chart.drawGroupBarChart()
  chart.drawLegend(cat_color)
  // cur.chart = chart
  chart.drawTitle(data.title)
}

function load_group_bar_chart_horizontal(data, cat_position, cat_color, quantity) {
  let chart = new CCQ(data, cat_position, cat_color, quantity, 'horizontal')
  chart.drawAxis()
  chart.drawGroupBarChart()
  chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_scatter_plot(data, cat_color, x, y) {
  let chart = new CQQ(data, cat_color, x, y)
  // console.log('load_scatter_plot')
  // console.log(data)
  chart.drawAxis()
  chart.drawLine()
  chart.drawScatterPlot()
  if (chart.countScatter())
    chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_scatter_plot_horizontal(data, cat_color, x, y) {
  let chart = new CQQ(data, cat_color, x, y, 'horizontal')
  chart.drawAxis()
  chart.drawLine()
  chart.drawScatterPlot()
  if (chart.countScatter())
    chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}


function load_scatter_line_plot(data, cat_color, x, y) {
  let chart = new CQQ(data, cat_color, x, y, 'vertical', 'line')
  chart.drawAxis()
  chart.drawLine()
  chart.drawScatterPlot()
  if (chart.countScatter())
    chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_scatter_line_plot_horizontal(data, cat_color, x, y) {
  let chart = new CQQ(data, cat_color, x, y, 'horizontal', 'line')
  chart.drawAxis()
  chart.drawLine()
  chart.drawScatterPlot()
  if (chart.countScatter())
    chart.drawLegend(cat_color)
  chart.drawTitle(data.title)
}

function load_pie_chart(data, cat_position, cat_color, quantity) {
  let chart = new CQ(data, cat_position, cat_color, quantity)
  chart.drawPieChart()
  chart.drawLegend('c0')
  chart.drawTitle(data.title)
}


// 读取整个配置文件，分别进行处理，

var program = require('commander');



var type = "file2dir"; // "file2file"
// type = ""

if (type === 'file2dir') {
  program
    .version('0.0.1')
    .option('-i, --input <string>', 'input file setting') // try_set.json
    .option('-o, --output_dir <string>', 'output file setting, dataset.json')
    .parse(process.argv);

  // console.log(program.cheese)

  input_file = program.input
  directory = program.output_dir

  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory);
  }

  var input_data = JSON.parse(fs.readFileSync(input_file));
  // console.log(input_data)
  data_number = input_data.length;

  for (var index_number = 0; index_number < data_number; index_number++) {
    // console.log("du i ge shuju: ", index_number)
    let current_setting = input_data[index_number]
    // 清除相应的svg，只保留一只

    deal_with_data(current_setting)

    d3.select(document.body).select("svg").attr('xmlns', 'http://www.w3.org/2000/svg')

    // console.log(document.body.outerHTML);
    fs.writeFile(directory + "/" + current_setting.filename, document.body.innerHTML, 'utf8', (err) => {
      if (err) throw err;
    })

    d3.select(document.body)
      .selectAll("svg")
      .remove()

    // console.log(index_number)

    // console.log(document.body.outerHTML);

  }
} else {

  input_file = "try_dir/scatter_0000.json"

  var input_data = JSON.parse(fs.readFileSync(input_file));
  current_setting = input_data
  d3.select(document.body).selectAll("svg").remove()
  // 清除相应的svg，只保留一只

  deal_with_data(current_setting)
  d3.select(document.body).select('svg')
    .attr('xmlns', 'http://www.w3.org/2000/svg')
  // console.log(document.body.innerHTML);
  fs.writeFile("output_try.svg", document.body.innerHTML, 'utf8', (err) => {
    if (err) throw err;
  })
}