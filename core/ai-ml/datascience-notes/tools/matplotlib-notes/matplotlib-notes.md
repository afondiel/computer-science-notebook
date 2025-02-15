# Matplotlib Notes

## Introduction to Matplotlib

- Most popular data viz library 
- Created by John Hunter(Neurobiologist)
- Initially created for EEG/ECoG Visualization Tool 
- Inspired from MATLAB 

## Matplotlib Architecture
``` 
+--------------------------+
| Scripting Layer(pyplot)  |
+--------------------------+
+--------------------------+
| Artist Layer(artist)     |
+--------------------------+
+--------------------------+
| Backend Layer(FigCanvas..|
+--------------------------+
```

### Backend Layer(FigCanvas, renderer, events ...)

1. FigureCanvas : matplotlib.backend_bases.FigureCanvas 
	- Encompasses the area onto which the figure is drawn
2. Renderer :  matplotlib.backend_bases.Renderer
	-  Knows how to draw on the FigureCanvas
3. Event : matplotlib.backend_bases.Event 
	-  Handles user inputs such as keyboards strokes and mouse clicks

### Artist Layer(artist)

- Comprised of one main object - *Artist*
	-  Know how to use the Renderer to draw on the Canvas
- Responsible for : Title, lines, tock labals, and images, all correspond to individuals Artist instances 
- 2 types of artist objects : 
	-  1. Primitive : Line2D, Rectangle, Circle, and Text 
	-  2. Composite : Axis, Tick, Axes, and Figure
- Each composite artist may contain other composite artists as well as primitive artists

### Scripting Layer

- Comprised mainly of pyplot, a scripting interface that is lighter that the Artist layer 
- Let's see how we can generate the same histogram of 10000 random values using the pyplot interface 

## Hands-On Matplotlib: Basic Plotting

- Support by differents env : Python scripts, iPython shell, web app & servers ... jupyter nb
- "Dynamic" ploting using BACKENDS : modify plot, costomize ...
- use magic functions(%) to execute backend objects 
	- magic function starts w/ % sign
- Some backends : 
	- %matplotlib inline : plot window within the browser and not in separeted window
	- %matplotlib notebook : allows to modify figure once is RENDERED !
- Matplotlib - PANDAS 
	- df.plot(kind="line")
	-  df["x"].plot(kind="hist")

## Dataset on Immigration to Canada
- Dataset : 
	- src : United Nations (45 countries)
	- annual data on the flows of international migrants
	-  migrants to canada dataset 

- Import the dataset with pandas
- to check the imported data : df.head, df.describe ...

## Line Plots
- continuos dataset  
- plot which displys information as series of data points called "markers" connected by the straight line segments
- using pandas dataframe, map function to create an iterative point per axis value 

## References 

- https://matplotlib.org/
- https://fr.wikipedia.org/wiki/Matplotlib
- aosabook.org/en/matplotlib.html 
