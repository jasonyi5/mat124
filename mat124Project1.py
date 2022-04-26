import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

## global things
aggieBlue = '#022851'
aggieGold = '#FFBF00'
class SIRFigures:
	def __init__(self):
		return
	def SIR():
		fig,ax = plt.subplots()
		S = ax.text(
	    0, 0, "Susceptible", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		I = ax.text(
	    5, 0, "Infected", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		R = ax.text(
	    10, 0, "Recovered", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))


		SI = ax.text(
	    2.6, 0, "Infection (β)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))
		IR = ax.text(
	    7.25, 0, "Recovery (γ)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))

		ax.set_xlim(0,10)
		ax.set_ylim(-1,1)

		ax.axis('off')
		return plt.show()

	def SIRandDeath():
		fig,ax = plt.subplots()
		S = ax.text(
	    0, 0, "Susceptible", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		I = ax.text(
	    5, 0.35, "Infected", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		R = ax.text(
	    10, 0.35, "Recovered", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))


		SI = ax.text(
	    2.6, 0.17, "Infection (β)", ha="center", va="center", rotation=20, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))
		IR = ax.text(
	    7.25, 0.35, "Recovery (γ)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))


		birth = ax.text(
	    0, 0.35, "Birth (α)", ha="center", va="center", rotation=270, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))
		death = ax.text(
	    10, 0, "Death (µ)", ha="center", va="center", rotation=270, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))


		ax.set_xlim(0,10)
		ax.set_ylim(-1,1)

		ax.axis('off')
		return plt.savefig('SIRwithDeath.png')

	def SIRandVac():
		fig,ax = plt.subplots()
		S = ax.text(
	    0, 0, "Susceptible", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		I = ax.text(
	    5, 0, "Infected", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		R = ax.text(
	    10, 0, "Recovered", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))


		SI = ax.text(
	    2.6, 0, "Infection (β)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))
		IR = ax.text(
	    7.25, 0, "Recovery (γ)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))

		vaccine = mpatches.FancyArrowPatch((0,0.15),(10,0.1),mutation_scale = 80,connectionstyle='arc3,rad=-0.3',fc='white',ec=aggieGold,lw=2)
		ax.add_patch(vaccine)

		SR = ax.text(
	    5, 0.51, "Vaccination (pN)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="round,pad=0.3", fc="None", ec='None', lw=2))

		ax.set_xlim(-0.1,10)
		ax.set_ylim(-1,1)

		ax.axis('off')
		return plt.show()

	def SIRandVacandDeath():
		fig,ax = plt.subplots()
		S = ax.text(
	    0, 0, "Susceptible", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		I = ax.text(
	    5, 0, "Infected", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))
		R = ax.text(
	    10, 0, "Recovered", ha="center", va="center", rotation=0, size=15,
	    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=aggieBlue, lw=2))


		SI = ax.text(
	    2.6, 0, "Infection (β)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))
		IR = ax.text(
	    7.25, 0, "Recovery (γ)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))

		vaccine = mpatches.FancyArrowPatch((0,0.15),(10,0.1),mutation_scale = 80,connectionstyle='arc3,rad=-0.3',fc='white',ec=aggieGold,lw=2)
		ax.add_patch(vaccine)

		SR = ax.text(
	    5, 0.51, "Vaccination (pN)", ha="center", va="center", rotation=0, size=10,
	    bbox=dict(boxstyle="round,pad=0.3", fc="None", ec='None', lw=2))

		birth = ax.text(
	    -1, 0.35, "Birth (α)", ha="center", va="center", rotation=270, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))
		death = ax.text(
	    10.9,0.35, "Death (µ)", ha="center", va="center", rotation=90, size=10,
	    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec=aggieGold, lw=2))

		ax.set_xlim(-0.1,10)
		ax.set_ylim(-1,1)

		ax.axis('off')
		return plt.show()

## PHASE PORTRAIT


class solveODE:
	def __init__(self):
		return
	def portrait(self,time,funct1,funct2,R,fig,ax):

		## where to place phase portrait
		y1 = np.linspace(0,90,R)
		y2 = np.linspace(0,6.5,R)

		Y1,Y2 = np.meshgrid(y1,y2)
		u,v = np.zeros(Y1.shape),np.zeros(Y2.shape)
		NI,NJ = Y1.shape
		for i in range(NI):
			for j in range(NJ):
				x = Y1[i,j]
				y = Y2[i,j]
				u[i,j] = funct1(time,x,y)
				v[i,j] = funct2(time,x,y)
		quiveropts = dict(color='k', headlength=0, pivot='middle', scale=1, 
	    linewidth=.5, units='xy', width=.05, headwidth=1)

		r = np.power(np.add(np.power(u,2),np.power(v,2)),0.5)
		Q = ax.quiver(Y1,Y2,u/r,v/r,headaxislength=4.5,) #**quiveropts) #np.arctan2(v,u),angles='xy',cmap='hsv',lw=5)

		## Code to make colorful phase portrait (unpreferred for report)
	#	Q = ax.quiver(Y1,Y2,u/r,v/r,np.arctan2(v,u),angles='xy',cmap='hsv',lw=5)

	#	fig.colorbar(Q,orientation = 'vertical')
	#	ax.set_facecolor('k')

		return

	def rungeKutta(self,a,b,N,y0,func):
		step = (b - a) * (1 / N)
		t = np.arange(a,b,step)
		yL = [y0]
		for i, time in np.ndenumerate(t):
			yi = yL[i[0]]
			y1 = func(time,yi)
			y2 = func((time + (step / 2)),(yi + ((step / 2) * y1)))
			y3 = func((time + (step / 2)),(yi + ((step / 2) * y2)))
			y4 = func((time + (step)),(yi + ((step) * y3)))
			y5 = yi + (step / 6) * (y1 + 2*y2 + 2*y3 + y4)
			yL.append(y5)
		yL.pop(-1)
		y = np.asarray(yL)
		return y
	def odYi45(self,a,b,N,y0,func):
		fig,ax = plt.subplots()
		step = (b - a) * (1 / N)
		time = np.arange(a,b,step)
		sol = []
		for i, initial in np.ndenumerate(y0):
			sol.append(self.rungeKutta(a,b,N,initial,func))
		for solution in sol:
			ax.plot(time,solution)
		ax.grid(True,which='both')
		ax.axhline(y=0,color='k',linestyle='solid')
		ax.axvline(x=0,color='k',linestyle='solid')
		return plt.show()
	def coupledRK(self,a,b,N,x0,y0,xFunc,yFunc):
		fig,ax = plt.subplots()
		step = (b - a) * (1 / N)
		t = np.arange(a,b,step)
		xL = [x0]
		yL = [y0]
		for i, time in np.ndenumerate(t):
			xi = xL[i[0]]
			yi = yL[i[0]]

			x1 = xFunc(time,xi,yi)
			y1 = yFunc(time,xi,yi)

			x2 = xFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)))
			y2 = yFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)))

			x3 = xFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)))
			y3 = yFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)))

			x4 = xFunc(time + step,xi + (step*x3),yi + (step*y3))
			y4 = yFunc(time + step,xi + (step*x3),yi + (step*y3))

			xL.append(xi + (step / 6)*(x1 + 2*x2 + 2*x3 + x4))
			yL.append(yi + (step / 6)*(y1 + 2*y2 + 2*y3 + y4))
		xL.pop(-1)
		yL.pop(-1)

		x = np.asarray(xL)
		y = np.asarray(yL)
		ax.plot(t,x)
		ax.plot(t,y)
		return plt.show()
	def coupledRKRange(self,a,b,N,x0,y0,xFunc,yFunc,ax,A,B,u):
		step = (b - a) * (1 / N)
		t = np.arange(a,b,step)
		xL = [x0]
		yL = [y0]
		for i, time in np.ndenumerate(t):
			xi = xL[i[0]]
			yi = yL[i[0]]

			x1 = xFunc(time,xi,yi)
			y1 = yFunc(time,xi,yi)

			x2 = xFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)))
			y2 = yFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)))

			x3 = xFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)))
			y3 = yFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)))

			x4 = xFunc(time + step,xi + (step*x3),yi + (step*y3))
			y4 = yFunc(time + step,xi + (step*x3),yi + (step*y3))

			xL.append(xi + (step / 6)*(x1 + 2*x2 + 2*x3 + x4))
			yL.append(yi + (step / 6)*(y1 + 2*y2 + 2*y3 + y4))
		xL.pop(-1)
		yL.pop(-1)

		x = np.asarray(xL)
		y = np.asarray(yL)

		return ax.plot(x,y,color='red')

	def coupledODYi45(self,a,b,N,x0,y0,xFunc,yFunc,A,B,g,u,p):
		fig,ax = plt.subplots()
		step = (b - a) * (1 / N)
		time = np.arange(a,b,step)
		sol = []
		for initialX,initialY in zip(x0,y0):
			sol.append(self.coupledRKRange(a,b,N,initialX,initialY,xFunc,yFunc,ax,A,B,u))

		for solution in sol:
			try:
				ax.plot(time,solution,color = aggieGold)
			except:
				pass
		ax.grid(True,which='both')
		ax.axhline(y=0,color='k',linestyle='solid')
		ax.axvline(x=0,color='k',linestyle='solid')

		## nullclines
		ax.axvline(x=((g + u) / B),color = aggieBlue,linestyle='dashed',lw=2)
		ax.axhline(y=0,color = aggieBlue,linestyle='dashed',lw=2)

		R = ((1 - p)*A) / (u * (g + u))

		ax.set_xlabel('Susceptible Population')
		ax.set_ylabel('Infected Population')

		xC = np.linspace(0,100,1000)
		ax.plot(((1 - p)*A) / ((B * xC) + u),xC,color = aggieBlue,linestyle='dashed',lw = 2)

		k1 = (g + u) / ((A * (1 - p))*B)
		ax.scatter((g + u) / B,(-u*k1 + 1) / (k1* B), color = 'cyan',marker ='*',s = 200,zorder=10)

		## phase portraint
		self.portrait(time,xFunc,yFunc,20,fig,ax)

		ax.set_xlim([0,91])
		ax.set_ylim([0,6.5])

		return plt.show()
	def tripleRKRange(self,a,b,N,x0,y0,z0,xFunc,yFunc,zFunc,ax):
		step = (b - a) * (1 / N)
		t = np.arange(a,b,step)
		xL = [x0]
		yL = [y0]
		zL = [z0]
		for i, time in np.ndenumerate(t):
			xi = xL[i[0]]
			yi = yL[i[0]]
			zi = zL[i[0]]

			x1 = xFunc(time,xi,yi,zi)
			y1 = yFunc(time,xi,yi,zi)
			z1 = zFunc(time,xi,yi,zi)

			x2 = xFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)),(zi + ((step / 2)*z1)))
			y2 = yFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)),(zi + ((step / 2)*z1)))
			z2 = zFunc(time + (step / 2),(xi + ((step / 2)*x1)),(yi + ((step / 2)*y1)),(zi + ((step / 2)*z1)))

			x3 = xFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)),(zi + ((step / 2)*z2)))
			y3 = yFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)),(zi + ((step / 2)*z2)))
			z3 = zFunc(time + (step / 2),(xi + ((step / 2)*x2)),(yi + ((step / 2)*y2)),(zi + ((step / 2)*z2)))

			x4 = xFunc(time + step,xi + (step*x3),yi + (step*y3),zi + (step*z3))
			y4 = yFunc(time + step,xi + (step*x3),yi + (step*y3),zi + (step*z3))
			z4 = zFunc(time + step,xi + (step*x3),yi + (step*y3),zi + (step*z3))

			xL.append(xi + (step / 6)*(x1 + 2*x2 + 2*x3 + x4))
			yL.append(yi + (step / 6)*(y1 + 2*y2 + 2*y3 + y4))
			zL.append(zi + (step / 6)*(z1 + 2*z2 + 2*z3 + z4))
		xL.pop(-1)
		yL.pop(-1)
		zL.pop(-1)

		x = np.asarray(xL)
		y = np.asarray(yL)
		z = np.asarray(zL)

		total  = x + y + z

		ax.plot(t,x,label = 'Susceptible')
		ax.plot(t,y,label = 'Infected')
		ax.plot(t,z,label = 'Recovered')
		ax.plot(t,total,label = 'Total Population')

		ax.legend()
		return
	def tripleODYi45(self,a,b,N,x0,y0,z0,xFunc,yFunc,zFunc):
		fig,ax = plt.subplots()
		step = (b - a) * (1 / N)
		time = np.arange(a,b,step)
		sol = []
		for i, initialX in np.ndenumerate(x0):
			for j, initialY in np.ndenumerate(y0):
				for k, initialZ in np.ndenumerate(z0):
					sol.append(self.tripleRKRange(a,b,N,initialX,initialY,initialZ,xFunc,yFunc,zFunc,ax))
		for solution in sol:
			try:
				ax.plot(time,solution)
			except:
				pass
		ax.grid(True,which='both')
		ax.axhline(y=0,color='k',linestyle='solid')
		ax.axvline(x=0,color='k',linestyle='solid')
		ax.set_ylim([0,1010])

		ax.set_xlabel('Time (days)')
		ax.set_ylabel('Population')
		return plt.show()
	def nullclineNoV(a,b,g,u):
		fig,ax = plt.subplots()

		## plot nullclines
		x = np.linspace(0.001,3,1000)
		y = a / (b*x + u)
		ax.plot(y,x,color=aggieBlue,linestyle='dashed')
		ax.axhline(0,color=aggieBlue,linestyle='dashed')
		ax.axvline((g + u) / b,color=aggieBlue,linestyle='dashed')

		## plot arrows on curved nullclines
		xA = np.linspace(0.001,3,10)
		yA = a / (b*xA + u)
		xdA = xA + 0.5
		ydA = a / (b*xdA + u)
		for yc,xc in zip(yA,xA):
			if yc < 0.9:
				ax.arrow(yc,xc,0,-0.15,length_includes_head=True,head_width=0.03,head_length=0.05)
			else:
				ax.arrow(yc,xc,0,0.15,length_includes_head=True,head_width=0.03,head_length=0.05)

		## arrows on verticle nullcline
		xc2 = (g + u) / b
		yc2 = np.linspace(0,3,10)
		for point in yc2:
			if point < 1:
				ax.arrow(xc2,point,0.15,0,length_includes_head=True,head_width=0.03,head_length=0.05)
			else:
				ax.arrow(xc2,point,-0.15,0,length_includes_head=True,head_width=0.03,head_length=0.05)

		xc3 = np.linspace(0,3,10)
		for point in xc3:
			if point < 1.70:
				ax.arrow(point,0,0.15,0,length_includes_head=True,head_width=0.03,head_length=0.05)
			if point > 1.70:
				ax.arrow(point,0,-0.15,0,length_includes_head=True,head_width=0.03,head_length=0.05)

		## annotate basic regions with A,B,C,D
		## A
		ax.arrow(2.25,1.5,-0.15,0.15,length_includes_head=True,head_width=0.1,head_length=0.05,color='red')

		## B
		ax.arrow(0.7,2.25,-0.15,-0.15,length_includes_head=True,head_width=0.1,head_length=0.05,color='red')

		## C
		ax.arrow(0.25,1.25,0.15,-0.15,length_includes_head=True,head_width=0.1,head_length=0.05,color ='red')

		## D
		ax.arrow(1,0.22,0.15,0.15,length_includes_head=True,head_width=0.1,head_length=0.05,color = 'red')

		R = (a*b) / (u * (g + u))
		ax.scatter(R,0,color='purple',marker = '*',zorder = 10, s= 300)

		k1 = (g + u) / (a*b)
		ax.scatter((g + u) / b,(-u*k1 + 1) / (k1* b), color = 'darkorange',marker = '*',zorder = 10,s = 300)

		ax.set_ylim([-0.05,3])
		ax.set_xlim([0,3])
		ax.set_ylabel('Infected Population')
		ax.set_xlabel('Susceptible Population')

		return plt.show()


class noVaccine:
	def __init__(self,a,b,g,u,N):
		self.a = a
		self.b = b
		self.g = g
		self.u = u
		self.N = N
		return
	def dSdt(self,t,S,I,R):
		return self.a - self.b*S*I - self.u*S
	def dIdt(self,t,S,I,R):
		return self.b*S*I - self.g*I - self.u*I
	def dRdt(self,t,S,I,R):
		return self.g*I - self.u*R
class vaccine:
	def __init__(self,a,b,g,u,N,p):
		self.a = a
		self.b = b
		self.g = g
		self.u = u
		self.N = N
		self.p = p
		return
	def dSdt(self,t,S,I):
		return (1 - self.p)*self.a -(self.b*S*I) - self.u*S
	def dIdt(self,t,S,I):
		return self.b*S*I - self.g*I - self.u*I
	def dRdt(self,t,S,I,R):
		return (self.p*self.a) + self.g*I - self.u*R
