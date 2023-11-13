### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d8e3d937-bcda-4c84-b543-e1324f696bbc
begin 
	import Pkg
	import Base
	using SpeedyWeather
	using NCDatasets
	using Printf, CairoMakie, PlutoUI, JLD2
	using LinearAlgebra, SparseArrays
	using DataDeps
	using Statistics: mean, norm

	
	html"""
	<div style="
	position: absolute;
	width: calc(100% - 30px);
	border: 50vw solid #282936;
	border-top: 200px solid #282936;
	border-bottom: none;
	box-sizing: content-box;
	left: calc(-50vw + 15px);
	top: -200px;
	height: 200px;
	pointer-events: none;
	"></div>
	
	<div style="
	height: 200px;
	width: 100%;
	background: #282936;
	color: #fff;
	padding-top: 68px;
	">
	<span style="
	font-family: Vollkorn, serif;
	font-weight: 700;
	font-feature-settings: 'lnum', 'pnum';
	"> <p style="
	font-size: 1.5rem;
	opacity: .8;
	"><em>Sections 2.3 and 2.4</em></p>
	<p style="text-align: center; font-size: 2rem;">
	<em> Climate models: solving the climate system in Julia </em>
	</p>
	
	<style>
	body {
	overflow-x: hidden;
	}
	</style>"""
end

# ╔═╡ 8ef88534-dac4-4a62-b623-dcaf63482a96
md"""
# Section 2.3: a latitude-dependent climate

##### What is a climate model?

A climate model is a complex PDE solver that solves a set of differential equations on a discretized version of the earth, with land, ocean, and atmosphere discretized on a three-dimensional grid. The equations usually ensure the conservation of mass, momentum, and energy.

$(Resource("https://d32ogoqmya1dw8.cloudfront.net/images/eet/envisioningclimatechange/gcm_grid_graphic.jpg", :height => 400))

**Figure**: schematic depicting the discretization in a general circulation model (GCM) \
Climate models are usually massive, sophisticated models which require years to develop and have to run on high-performance computing centers.

##### Improving our first climate model
In section 2.1 you have seen your first climate model, a system of equations that predicts the earth's average surface temperature depending on the sun's forcing and the absorption of the atmosphere. 

```math
\begin{align}
C_a \frac{d T_a}{dt} & = \varepsilon \sigma T_s ^4 - 2\varepsilon \sigma T_a^4 \\
C_s \frac{d T_s}{dt} & = \varepsilon \sigma T_a^4 - \sigma T_s ^4 + (1 - \alpha) \frac{S_0}{4}
\end{align}
```

$(Resource("https://scied.ucar.edu/sites/default/files/styles/extra_large/public/images/earth_energy_budget_percent_nasa.gif.webp?itok=w6dUIlIm", :height => 300))
**Figure**: Earth's energy budget

This simple model is handy for predicting global heating and cooling but does not bring us much further. In order to characterize our climate, an essential quantity we are interested in predicting is the temperature difference between low and high latitudes. The latitudinal temperature gradient is a significant quantity that drives motions in the atmosphere and is the cause of all major climatic events. We will also later see that the latitudinal temperature gradient is one measure of the efficiency of the global climate system in redistributing heat and is used to test the ability of models to represent the climate system through time

To improve our simple model, we will introduce an extra dimension, the latitude


$(Resource("https://raw.githubusercontent.com/simone-silvestri/ComputationalThinking/main/two-models.png", :height => 200))

**Figure**: difference between a 0D model (averaged over earth's surface) and a 1D model (averaged over spherical segments)

"""

# ╔═╡ cfb8f979-37ca-40ab-8d3c-0053911717e7
md"""
## Variable insolation

Let us introduce some variability in our climate system. The variability is imposed on the system by the forcing. You already saw that the spatially and annual averaged radiative flux that reaches the earth (in units of W/m²) is
```math
S_0 / 4 , \ \ \ \ \text{with} \ \ \ \ S_0 \approx 1365 \ W m^{-2}
```
of which only ``(1 - \alpha)`` is absorbed (where ``\alpha`` is the albedo (or reflectivity) of the surface). This flux is not distributed equally along the surface of the planet. The insolation amount and intensity vary in different locations, days, seasons, and years. \

Three main parameters affect the intensity of the incoming solar radiation:

- the latitude
- the hour of the day
- the day of the year

In practice, we will simplify the system by averaging the dependencies on hour and day. What remains is a 1D model which depends on time and latitude.
"""

# ╔═╡ eb95e773-b12a-40a4-a4f1-9dced86fc8a2
md"""
##### Latitudinal dependency (angle ``\phi``)

Different parts of Earth’s surface receive different amounts of sunlight (Figure below). The Sun’s rays strike Earth’s surface most directly at the equator. This focuses the rays on a small area.  Near the poles, the Sun’s rays strike the surface at a slant. This spreads the rays over a wide area.  The more focused the rays are, the more energy an area receives, and the warmer it is.

$(Resource("https://static.manitobacooperator.ca/wp-content/uploads/2020/02/18151642/insolation_CMYK.jpg#_ga=2.245013061.1375356746.1664813564-1302273094.1664813564", :height => 300))

##### Hourly dependency (angle ``h``)

As the earth rotates along its axis, the same happens in the east-west direction. At noon, the rays will be parallel to the earth, facing the smallest surface area. In the evening/morning, rays are slanted, facing a larger surface area. We can express this dependency as an angle (``h``) that takes the value of 0 at noon, positive values in the afternoon, and negative values in the morning. Since the Earth rotates 15° per hour, each hour away from noon corresponds to an angular motion of the sun in the sky of 15°

##### Seasonal dependency (angle ``\delta``)

The declination of the sun is the angle between the equator and a line drawn from the center of the Earth to the center of the sun. It is positive when it is north and negative when it is south. The declination reaches its maximum value, +23° 17', on 21 June (the summer solstice in the northern hemisphere, the winter solstice in the southern hemisphere). The minimum value, −23° 27', is reached on 20 December. 
[Animation showing the declination angle.](https://www.pveducation.org/sites/default/files/PVCDROM/Properties-of-Sunlight/Animations/earth-rotation/earth-rotation_HTML5.html)
The declination, in degrees, for any given day may be calculated (approximately) with the equation:

```math
\delta = 23.45^\circ \sin{\left(\frac{360}{365.25} \cdot (\text{day} - 81)\right)}
```

where ``\text{day}`` starts from 1 on the 1st of January and ends at 365 on December 31st, while the 81st day is the spring equinox (22nd of March), where the earth's axis is perpendicular to the orbit



$(Resource("https://ars.els-cdn.com/content/image/3-s2.0-B9780080247441500061-f01-03-9780080247441.gif", :height => 300))
**Figure**: Declination angle (``\delta``) versus days after the equinox. 
"""

# ╔═╡ 75cacd05-c9f8-44ba-a0ce-8cde93fc8b85
md"""
#### Bringing it all together

$(Resource("https://raw.githubusercontent.com/simone-silvestri/ComputationalThinking/main/angles.png", :height => 300))
**Figure:** Angles that define solar flux with respect to earth)

We model the instantaneous solar flux with
```math
Q \approx S_0 \left( \underbrace{\sin{\phi} \sin{\delta} + \cos{h} \cos{\delta} \cos{\phi}}_{\cos{\theta_z}} \right)
```

where ``\theta_z`` is the zenith angle, shown in the figure below

$(Resource("https://ars.els-cdn.com/content/image/3-s2.0-B9780128121498000028-f02-02-9780128121498.jpg", :height => 300))
**Figure:** Zenith angle, (M. Rosa-Clot & G. Tina, Submerged and Floating Photovoltaic Systems, 2018, chapter 2)

The cosine of the zenith angle is the useful percentage of ``S_0`` which strikes the earth's surface.
What does the first term on the right-hand side express? And the second?
Negative insolation does not exist... so negative values of ``\cos{\theta_z}`` indicate night-time, for which ``Q=0``. When ``cos(\theta_z)`` is exactly equal to zero, we are at sunset or sunrise.
We can calculate the sunset (and sunrise) hour angle (``h_{ss}``) as follows
```math
\cos{h_{ss}} = - \tan{\phi}\tan{\delta}
```

##### Polar Sunrise and Sunset

Due to the inclination of the earth's axis, some regions experience days and nights that extend beyond 24 hours. This phenomenon is called Polar day and Polar night. The longest days and nights are at -90/90 ᵒN (the poles), which experience a single day and night in the year. Polar sunrise and sunset occur at a latitude that satisfies
```math
|\phi| > 90ᵒ - |\delta|
```
``\delta`` and ``\phi`` of the same sign mean that the sun is rising, vice-versa, if the signs are opposite the sun is setting

##### Daily Insolation

Let's calculate the daily insolation (in 24 hr). Since we express the day in ``2\pi`` radians and ``Q = 0`` if ``|h| > h_{ss}``

```math
\langle Q \rangle_{day}  = \frac{S_0}{2\pi} \int_{-h_{ss}}^{h_{ss}} (\sin{\phi}\sin{\delta} + \cos{\phi}\cos{\delta}\cos{h} ) \ dh
```

which is easily integrated to 

```math
\langle Q \rangle_{day}  = \frac{S_0}{\pi} \left( h_{ss}\sin{\phi}\sin{\delta}  + \cos{\phi}\cos{\delta}\sin{h_{ss}} \right)
```

"""


# ╔═╡ 18ddf155-f9bc-4e5b-97dc-762fa83c9931
function daily_insolation(lat; day = 81, S₀ = 1365.2)

	march_first = 81.0
	ϕ = deg2rad(lat)
	δ = deg2rad(23.45) * sind(360*(day - march_first) / 365.25)

	h₀ = abs(δ) + abs(ϕ) < π/2 ? # there is a sunset/sunrise
		 acos(-tan(ϕ) * tan(δ)) :
		 ϕ * δ > 0 ? π : 0.0 # all day or all night
		
	# Zenith angle corresponding to the average daily insolation
	cosθₛ = h₀*sin(ϕ)*sin(δ) + cos(ϕ)*cos(δ)*sin(h₀)
	
	Q = S₀/π * cosθₛ 

	return Q
end

# ╔═╡ 87fdc7c2-536e-4aa1-9f68-8aec4bc7069d
md""" day $(@bind day_in_year PlutoUI.Slider(1:365, show_value=true)) """

# ╔═╡ 8d4d8b93-ebfe-41ff-8b9e-f8931a9e83c2
begin
	latitude = -90:90
	δ = (23 + 27/60) * sind(360*(day_in_year - 81.0) / 365.25)
	
	polar_ϕ = 90 - abs(δ)

	function day_to_date(day)
		months = (:Jan, :Feb, :Mar, :Apr, :May, :Jun, :Jul, :Aug, :Sep, :Oct, :Nov, :Dec)
		days_in_months   = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 31, 31)
		days_till_months = [1, [sum(days_in_months[1:i]) for i in 1:11]...]

		month = searchsortedlast(days_till_months, day)
		day_in_month = month > 1 ? day - sum(days_in_months[1:month-1]) : day
		
		return "$(day_in_month) " * string(months[month])
	end
	
	Q = daily_insolation.(latitude; day = day_in_year)
	fig = Figure(resolution = (700, 300))
	ax = Axis(fig[1, 1], title = "Daily average insolation, Day: $(day_in_year) ($(day_to_date(day_in_year)))", xlabel = "latitude ᵒN", ylabel ="daily ⟨Q⟩ Wm⁻²")
	lines!(ax, latitude, Q, linewidth = 2, color=:red)
	lines!(ax,  polar_ϕ*[1, 1], [0, 600], linestyle=:dash, color=:yellow)
	lines!(ax, -polar_ϕ*[1, 1], [0, 600], linestyle=:dash, color=:yellow)
	ax.xticks = [-90, -90+23, -25, -50, 0, 25, 50, 90-23, 90]
	ylims!(ax, -10, 600)
	
	md"""
	$(current_figure())
	**Figure**: daily average insolation in Wm⁻². The yellow lines correspond to the latitude of polar sunset (sunrise), above (below) which it is only night (day) 

	$(Resource("https://ars.els-cdn.com/content/image/1-s2.0-S0074614202800171-gr8.jpg", :height => 400))
	**Figure**: Daily mean solar insolation (Q/24 hr) as a function of latitude and day of year in units of W m−2 based on a solar constant of 1366 W m−2. The shaded areas denote zero insolation. The position of vernal equinox (VE), summer solstice (SS), autumnal equinox (AE), and winter solstice (WS) are indicated with solid vertical lines. Solar declination is shown with a dashed line (K.N. Liou, An Introduction to Atmospheric Radiation, 2002, chapter 2)
	"""
end

# ╔═╡ 25223f7b-22f7-46c2-9270-4430eb6c186e
begin
	function annual_mean_insolation(lat; S₀ = 1365.2)
		Q_avg = 0
		for day in 1:365
			Q_avg += daily_insolation(lat; day, S₀) / 365
		end

		return Q_avg
	end
	
	Q_avg = zeros(length(-90:90))
	for (idx, lat) in enumerate(-90:90)
		Q_avg[idx] += annual_mean_insolation(lat)
	end

	fm = Figure(resolution = (700, 300))
	am = Axis(fm[1, 1], title = "Yearly average insolation", xlabel = "latitude ᵒN", ylabel ="yearly ⟨Q⟩ Wm⁻²")
	lines!(am, -90:90, Q_avg)

	md"""
	For our purposes, let's forget about the seasons and calculate the yearly average mean isolation

	```math
	\langle Q \rangle_{\text{yr}} = \frac{1}{365}\sum_{\text{day} = 1}^{365} \langle Q \rangle_{\text{day}}
	```
	
	$(current_figure())
	**Figure**: Annual mean insolation
	
	We can see that the average insolation is much lower (~2.5X) at the poles compared to the equator! This is reassuring given the climate we experience!
	
	"""
end


# ╔═╡ 034fc483-b188-4b2a-891a-61b76c74072d
md"""
## Solving the climate system: equilibrium solution

Let us recall the system of ODE that governs the surface and atmospheric temperature. \
The forcing is now be latitude-dependent, resulting in a latitude-dependent temperature

```math
\begin{align}
C_a \frac{d T_a}{dt} & = \varepsilon \sigma T_s ^4 - 2\varepsilon \sigma T_a^4 \\
C_s \frac{d T_s}{dt} & = \varepsilon \sigma T_a^4 - \sigma T_s ^4 + (1 - \alpha)  Q(\phi)
\end{align}
```
Here ``T_s`` is the surface (or ocean) temperature, ``T_a`` is the atmospheric temperature, ``\varepsilon`` in the emissivity of the atmosphere, ``\alpha`` is the earth's albedo and ``C_s`` and ``C_a`` are the heat capacities of the surface and atmosphere, respectively. ``Q(\phi)`` is the yearly averaged latitudinal insolation.  When the system reaches equilibrium it stops evolving in time \
(i.e the ``dT/dt = 0``)
```math
\begin{align}
& \varepsilon \sigma T_{E,s}^4 - 2\varepsilon \sigma T_{E,a}^4 = 0 \\
& \varepsilon \sigma T_{E,a}^4 - \sigma T_{E,s}^4 + (1 - \alpha) Q(\phi) = 0
\end{align}
```

which yields
```math
\begin{align}
& T_{E,a}(\phi) = \sqrt[4]{\frac{(1 - \alpha) Q (\phi)}{\sigma (2 - \varepsilon)}} \\
& T_{E,s}(\phi) = \sqrt[4]{\frac{2(1 - \alpha) Q (\phi)}{\sigma (2 - \varepsilon)}}
\end{align}
```
"""

# ╔═╡ 5d31e2a8-e357-4479-bc48-de1a1b8bc4d4
md"""
## Solving the climate system: numerical solution

The equilibrium solution is a good approximation, but the climate is always evolving, so being able to solve the time-dependent system is important for climate predictions. The system is too complicated to be solved analytically, but we can easily solve it numerically. A numerical solution of a differential equation is an approximation of the analytical solution usually computed by _discretizing_ the derivatives.

Let's see how to do this with two different methods:
- explicit time stepping
- implicit time stepping

We first assume that the time derivative can be written simply as
```math
\frac{d T}{dt} \approx \frac{T^{(n+1)} - T^{(n)}}{t^{(n+1)} - t^{(n)}}
```
where the ``n`` superscript stands for the time instant and the time step ``\Delta t`` is defined as ``\Delta t = t^{(n+1)} - t^{(n)}``

Now we can rewrite the equations as
```math 
\begin{align}
C_a \frac{T_a^{(n+1)} -  T_a^{(n)}}{\Delta t} & = G_a \\
C_s \frac{T_s^{(n+1)} -  T_s^{(n)}}{\Delta t} & = G_s 
\end{align}
```
where ``G`` are the _tendency terms_ defined as
```math
\begin{align} 
G_a & = \varepsilon \sigma T_s ^4 - 2\varepsilon \sigma T_a^4  \\ 
G_s & = \varepsilon \sigma T_a^4 - \sigma T_s ^4 + F 
\end{align}
```
and ``F`` is the _forcing_
```math
F = (1-\alpha) Q
```

#### Explicit time stepping

\

The tendencies are calculated at time ``n``, so the update rule becomes:
```math 
\begin{align}
T_a^{(n+1)} & = T_a^{(n)} + \frac{\Delta t}{C_a} G_a^{(n)} \\
T_s^{(n+1)} & = T_s^{(n)} + \frac{\Delta t}{C_s} G_s^{(n)} 
\end{align}
```
It is called _explicit time stepping_ because the values of ``G_a^{(n)}`` and ``G_s^{(n)}`` are readily available and the update rule for the time step ``n+1`` is explicitly dependent on time step ``n``.
Explicit time stepping is fast and simple to implement, but it has some shortcomings when time stepping with large ``\Delta t``

#### Implicit time stepping

\

The tendencies are calculated at time ``n+1``. This means that they are not readily available and we have to relate them to temperatures at time ``n+1``. Then
```math
\begin{align} 
G_a^{(n+1)} & = \varepsilon \sigma \left(T_s^{(n+1)}\right) ^4 - 2\varepsilon \sigma \left(T_a^{(n+1)}\right)^4 \\ 
G_s^{(n+1)} & = \varepsilon \sigma \left(T_a^{(n+1)}\right)^4 - \sigma \left(T_s^{(n+1)}\right)^4 + F
\end{align}
```
We would like to express the ODEs as a linear system, but these equations are non-linear. \
Fortunately, if we assume that the temperature does not change significantly in one-time step (``T^{(n+1)} - T^{(n)} \ll T^{(n)}``) we can linearize ``\left(T^{(n+1)}\right)^4`` as
```math
\left(T^{(n+1)}\right)^4 \approx \left(T^{(n)}\right)^3 T^{(n+1)}
```

```math
\begin{align} 
G_a^{(n+1)} & = \varepsilon \sigma \left(T_s^{(n)}\right)^3 T_s^{(n+1)} - 2\varepsilon \sigma \left(T_a^{(n)}\right)^3 T_a^{(n+1)} \\ 
G_s^{(n+1)} & = \varepsilon \sigma \left(T_a^{(n)}\right)^3 T_a^{(n+1)} - \sigma \left(T_s^{(n)}\right)^3 T_s^{(n+1)} + F
\end{align}
```

Substituting the expressions for the tendencies in the update equations we get
```math
\begin{align}
\left(C_a + \Delta t 2 \varepsilon \sigma \left(T_a^{(n)}\right)^3\right) T_a^{(n+1)} - \Delta t \varepsilon \sigma \left(T_s^{(n)}\right)^3 T_s^{(n+1)} & = C_a T_a^{(n)}\\
\left( C_s  + \Delta t \sigma\left(T_s^{(n)}\right)^3 \right) T_s^{(n+1)} - \Delta t \varepsilon \sigma \left(T_a^{(n)}\right)^3 T_a^{(n+1)} & = C_s T_s^{(n)} + \Delta t F
\end{align}
```

This is a system of linear equations in the variables ``T = [T_a^{(n+1)}``; ``T_s^{(n+1)}]`` representable as the linear system
```math
A T = b
```
where the matrix ``A`` is
```math
A = \begin{bmatrix}
C_a  + \Delta t 2 \varepsilon \sigma \left(T_a^{(n)}\right)^3 & - \Delta t\varepsilon \sigma \left(T_s^{(n)}\right)^3 \\
 - \Delta t  \varepsilon \sigma \left(T_a^{(n)}\right)^3  &  
C_s  + \Delta t \sigma\left(T_s^{(n)}\right)^3 \\ 
\end{bmatrix}
```
and the right hand side (``b``) is
```math
b = 
\begin{bmatrix}
C_a T_a^{(n)} \\ C_s T_s^{(n)} + \Delta t F
\end{bmatrix}
```

"""

# ╔═╡ 724901e9-a19a-4d5f-aa6a-79ec0f230f24
md"""
# Let's code our model in Julia

We can start creating a ```struct``` that contains the information we need, i.e., the parameters, the state, and the solution method of the system.

Some comments: 
- Temperature (and forcing) are vectors depending on the discrete latitudinal grid ``\phi``
- To retrieve parameters of the `struct` it is useful to write functions that we can later extend
- It is convenient to write a constructor with some default values and a `show` method 
"""

# ╔═╡ 15dee5d8-e995-4e5a-aceb-48bcce42e76d
md"""
### Coding an explicit time stepping function

Now we can write a function which evolves our model of a time step ``\Delta t``.
"""

# ╔═╡ 2287bff1-6fb0-4431-8e15-aff3d7b6e005
md"""
### Coding an implicit time stepping function

If we want to time step implicitly we have to solve the ``AT=b`` linear system \
Fortunately in Julia, solving a linear system is as simple as writing
```
T = A \ b
```

the only tricky part remaining is to construct the matrix \
Since temperature can be vectors, we align them starting from the surface temperature and following with the atmospheric temperature. Let us imagine we have three different latitudes (−45,0,45) where subscript refers to ``\phi = -45``, while 2 and 3 to ``0`` and ``45``. We can arrange our matrix in the following way
```math
  \begin{bmatrix}
    {D_a}_{1} & & & {d_a}_1 & &\\
    & {D_a}_{2} & & & {d_a}_2 & \\
	& & {D_a}_{3}  & & & {d_a}_2 \\
	{d_s}_1 & & & {D_s}_{1} & & \\
    & {d_s}_2 & & & {D_s}_{2} & \\
    & & {d_s}_3 & & & {D_s}_{3}  \\
  \end{bmatrix} \begin{bmatrix}
{T_a}_1^{n+1} \\ {T_a}_2^{n+1} \\ {T_a}_3^{n+1} \\ {T_s}_1^{n+1} \\ {T_s}_2^{n+1} \\ {T_s}_3^{n+1}
\end{bmatrix} = 
\begin{bmatrix}
C_a {T_a}_1^{n} \\ C_a {T_a}_2^{n} \\ C_a {T_a}_3^{n} \\ C_s {T_s}_1^{n} + \Delta t F_1  \\ C_s {T_s}_2^{n} + \Delta t F_2  \\ C_s {T_s}_3^{n} + \Delta t F_3
\end{bmatrix} 

```
where the diagonal terms are the sink terms, while the off-diagonal are the interexchange terms between surface and atmosphere. (Following the A matrix outlined above)
"""

# ╔═╡ e24e54a7-804e-40e8-818e-8766e5e3732b
md"""
Implicit time stepping implies constructing the matrix, calculating the rhs and solving the linear system
"""

# ╔═╡ 049e2164-24ac-467c-9d96-77510ac6ff57
md"""
### Model verification

Let's verify that our model reaches equilibrium with both implicit and explicit time stepping.

Some constants to be defined:
- the stefan Boltzmann constant (σ) in [Wm⁻²K⁻⁴] -> 5.67e-8
- the oceanic and atmospheric heat capacity in [Wm⁻²K⁻¹⋅day]

Note that we need the heat capacity in those units to be able to time step in [days].
"""

# ╔═╡ 69531da2-5b25-453b-bc86-2db6a944e62a
begin
	const σ  = 5.67e-8;
	const Cₛ = 1000.0 * 4000.0 * 4000 / (3600 * 24); #  ρ * c * H / seconds_per_day
	const Cₐ = 1e5 / 10 * 1000 / (3600 * 24);       # Δp / g * c / seconds_per_day
end

# ╔═╡ 039ec632-d238-4e63-81fc-a3225ccd2aee
equilibrium_temperature(lat, ε, α) =  
                 (2 * annual_mean_insolation(lat) * (1 - α) / (2 - ε) / σ)^(1/4)

# ╔═╡ 1431b11f-7838-41da-92e3-bcca9f4215b3
begin 
	# Types that specify the time-stepping method
	struct ExplicitTimeStep end
	struct ImplicitTimeStep end
	
	struct Model{S, T, K, E, A, F, C, ΦF, ΦC}
		stepper :: S  # time stepper with auxiliary variables
		Tₛ 		:: T  # surface temperature
		Tₐ 		:: T  # atmospheric temperature
		κ  		:: K  # thermal conductivity of the climate system
		ε  		:: E  # atmospheric emissivity
		α  		:: A  # surface albedo
		Q  		:: F  # forcing
		Cₛ 		:: C  # surface heat capacity
		Cₐ 		:: C  # atmospheric heat capacity
		ϕᶠ 		:: ΦF # the latitudinal grid at interface points (in radians)
		ϕᶜ 		:: ΦC # the latitudinal grid at center points (in radians)
	end

	Base.eltype(model::Model) = eltype(model.Tₛ)
	
	# convenience alias for dispatch
	const ExplicitModel = Model{<:ExplicitTimeStep}
	const ImplicitModel = Model{<:ImplicitTimeStep}
	
	# We define a constructor for the Model
	function Model(step = ImplicitTimeStep(), FT = Float32; 
				   ε = FT(0.8), 
				   α = FT(0.2985), 
				   κ = nothing, 
			   	   N = 45,
			   	   Cₛ = FT(Cₛ),
			   	   Cₐ = FT(Cₐ),
			   	   Q = annual_mean_insolation) 

		ϕᶠ = FT.(range(-π/2, π/2, length=N+1))
		ϕᶜ = (ϕᶠ[2:end] .+ ϕᶠ[1:end-1]) ./ 2
		Tₛ = FT(280.0) * ones(FT, N)
		Tₐ = FT(225.0) * ones(FT, N)
		Q = regularize_forcing(Q, ϕᶜ)

		return Model(step, Tₛ, Tₐ, κ, ε, α, Q, Cₛ, Cₐ, ϕᶠ, ϕᶜ)
	end

	regularize_forcing(Q, ϕ) = Q
	regularize_forcing(Q::Function, ϕ::Number) = eltype(ϕ)(Q(ϕ  * 180 / π))
	regularize_forcing(Q::Function, ϕ::AbstractArray) = eltype(ϕ).(Q.(ϕ .* 180 / π))

	# A pretty show method that displays the model's parameters
	function Base.show(io::IO, model::Model)
		print(io, "One-D energy budget model with:", '\n',
		"├── time stepping: $(timestepping(model))", '\n',
		"├── ε: ", show_parameter(emissivity(model)), '\n',
		"├── α: ", show_parameter(albedo(model)), '\n',
		"├── κ: ", show_parameter(diffusivity(model)), '\n',
		"└── Q: ", show_parameter(model.Q), " Wm⁻²")
	end
	
	# Let's define functions to retrieve the properties of the model.
	# It is always useful to define functions to extract struct properties so we 
	# have the possibility to extend them in the future
	# emissivity and albedo
	show_parameter(::Nothing)        = @sprintf("not active")
	show_parameter(p::Number)        = @sprintf("%.3f", p)
	show_parameter(p::AbstractArray) = @sprintf("extrema (%.3f, %.3f)", maximum(p), minimum(p))

	# Utility functions to @show the model
	timestepping(model::ExplicitModel) = "Explicit"
	timestepping(model::ImplicitModel) = "Implicit"

	# We define, again, the emissivities and albedo as function of the model
	emissivity(model) = model.ε
	emissivity(model::Model{<:Any, <:Any, <:Any, <:Function}) = model.ε(model)

	albedo(model) = model.α
	albedo(model::Model{<:Any, <:Any, <:Any, <:Any, <:Function}) = model.α(model)

	diffusivity(model) = model.κ
	diffusivity(model::Model{<:Any, <:Any, <:Function}) = model.κ(model)

	ImplicitRadiativeModel = Model{<:ImplicitTimeStep, <:Any, <:Nothing}
	ExplicitRadiativeModel = Model{<:ExplicitTimeStep, <:Any, <:Nothing}
end

# ╔═╡ de5d415f-8216-473d-8e0b-a73139540e1e
# Let's test the constructor and the show method
Model()

# ╔═╡ af022b36-34a9-497f-8b23-b76f5a98e741
@inline function tendencies(model::ExplicitRadiativeModel)
	# Temperatures at time step n
	Tₛ = model.Tₛ
	Tₐ = model.Tₐ
	
	# properties
	α  = albedo(model)
	ε  = emissivity(model)

	Gₛ = @. (1 - α) * model.Q + σ * (ε * Tₐ^4 - Tₛ^4)
	Gₐ = @. σ * ε * (Tₛ^4 - 2 * Tₐ^4)
	
	return Gₛ, Gₐ
end

# ╔═╡ c0ff6c61-c4be-462b-a91c-0ee1395ef584
@inline function construct_radiative_matrix(model, Δt)
	# Temperatures at time step n
	Tₛ = model.Tₛ
	Tₐ = model.Tₐ

	ε = emissivity(model)

	Cₐ = model.Cₐ
	Cₛ = model.Cₛ

	m = length(Tₛ)
	
	eₐ = @. Δt * σ * Tₐ^3 * ε
	eₛ = @. Δt * σ * Tₛ^3

	# We build and insert the diagonal entries
	Da = @. Cₐ + 2 * eₐ
	Ds = @. Cₛ + eₛ
	
	D  = vcat(Da, Ds)

	# the off-diagonal entries corresponding to the interexchange terms
	da = @. -ε * eₛ
	ds = @. -eₐ
	
	# spdiagm(idx => vector) constructs a sparse matrix 
	# with vector `vec` at the `idx`th diagonal 
	A = spdiagm(0 => D,
				m => da,
			   -m => ds)
	return A
end

# ╔═╡ 97e1ce89-f796-4bd1-8e82-94fc838829a6
begin
	# We only have radiation in our model for the moment...
	@inline construct_matrix(model::ImplicitRadiativeModel, Δt) = 
			construct_radiative_matrix(model, Δt)
	
	@inline function time_step!(model, Δt)
		# Construct the LHS matrix
		A = construct_matrix(model, Δt)

		α = albedo(model)

		# Calculate the RHS
		rhsₐ = @. model.Cₐ * model.Tₐ
		rhsₛ = @. model.Cₛ * model.Tₛ + Δt * (1 - α) * model.Q

		rhs = [rhsₐ..., rhsₛ...]

		# Solve the linear system
		T = A \ rhs

		nₐ = length(model.Tₐ)
		nₛ = length(model.Tₛ)

		@inbounds @. model.Tₐ .= T[1:nₐ]
		@inbounds @. model.Tₛ .= T[nₐ+1:nₐ+nₛ]

    	return nothing
	end
end

# ╔═╡ 191fa774-b5f2-41c5-b913-04a4b4138af3
md"""
Let's define some utility functions: 
- a function that evolves our model in time
- a function that plots the results as a function of ϕ
"""

# ╔═╡ 5da6b1c1-4a26-4159-9386-7de456c1c697
function plot_latitudinal_variables!(ϕ, vars; 
									 labels  = ["" for i in 1:length(vars)],
									 colors  = [:red for i in 1:length(vars)],
									 styles  = [:solid for i in 1:length(vars)],
									 widths  = [2 for i in 1:length(vars)],
									 ylabel  = "Temperature [ᵒC]", 
									 ylims   = nothing, 
									 title   = "", 
									 leg_pos = :cb,
									 ax_pos  = [1, 1],
									 res     = (700, 350),
									 fig     = Figure(resolution = res))

	axis = Axis(fig[ax_pos...]; title, ylabel, xlabel = "latitude [ᵒN]")	
	for (var, label, color, linestyle, linewidth) in zip(vars, labels, colors, styles, widths)
		lines!(axis, ϕ, var; linestyle, label, color, linewidth)
	end
	axislegend(axis, position = leg_pos, framevisible = false)
	!isnothing(ylims) && ylims!(axis, ylims)
	
	return fig
end

# ╔═╡ b85fdf41-ef8f-4314-bc3c-383947b9f02c
@bind values PlutoUI.combine() do Child
	md"""
	What happens ad different latitudes (``\phi``)? \
	And if we change ``\varepsilon`` or ``\alpha``? \
	And if we increase our Δt? (hint: try increasing Δt with high ``\varepsilon`` and low ``\alpha``)

	
	``\varepsilon`` $(
		Child(PlutoUI.Slider(0.05:0.05:1, show_value=true, default=0.75))
	) \
	
	``\alpha`` $(
		Child(PlutoUI.Slider(0.05:0.05:1, show_value=true, default=0.3))
	) \
	
	``\Delta t`` $(
		Child(PlutoUI.Slider(30:5:100, show_value=true, default=20))
	)

	"""
end


# ╔═╡ 00776863-2260-48a8-83c1-3f2696f11d96
begin 
	function compare_methods!(ε, α, Δt)

		# Construct the two models
		explicit_model = Model(ExplicitTimeStep(); α, ε)
		implicit_model = Model(ImplicitTimeStep(); α, ε)

		ϕ = explicit_model.ϕᶜ .* 180 ./ π
		
		# Time stepping parameters
		evolve_model!(explicit_model; Δt, stop_year = 800)
		evolve_model!(implicit_model; Δt, stop_year = 800)
		
		# Calculate equilibrium temperature analytically
		Tₛ_equilibrium = equilibrium_temperature.(ϕ, ε, α)
		Tₐ_equilibrium = Tₛ_equilibrium ./ 2^(1/4)

		title = @sprintf("final mean T: (T_eq, T_exp, T_imp) = (%.2f, %.2f, %.2f) K", mean(Tₛ_equilibrium), mean(explicit_model.Tₛ), mean(implicit_model.Tₛ))

		
		plot_latitudinal_variables!(ϕ, [Tₛ_equilibrium, explicit_model.Tₛ, implicit_model.Tₛ,
										Tₐ_equilibrium, explicit_model.Tₐ, implicit_model.Tₐ];
								   colors = [:skyblue1, :blue, :dodgerblue, :coral1, :red, 						 :firebrick1],
								   widths = [4, 2, 2, 4, 2, 2],
								   labels = ["equilibrium Ts", "explicit Ts", "implicit Ts",
								   			 "equilibrium Ta", "explicit Ta", "implicit Ta"],
								   styles = [:solid, :dash, :dashdot, :solid, :dash, :dashdot],
								   ylims  = (100, 350)
								   ylabel = "Temperature [K]")
		
		
		return nothing
	end
	
	compare_methods!(values[1], values[2], values[3])
	current_figure()
end

# ╔═╡ 16ca594c-c9db-4528-aa65-bab12cb6e22a
md"""
## Stability of time stepping methods

Temperature starts oscillating and then explodes when using a large Δt, this is because of the intrinsic stability of the time stepping method. A method is considered _unstable_ when it leads to unbounded growth despite the stability of the underlying differential equation. 
Let's analyze this by simplifying a bit our discretized atmospheric equation. We remove the coupling between the surface and the atmosphere. This is like saying that all of a sudden the atmosphere becomes transparent to the radiation coming from the earth (unlikely)
```math
C_a \frac{T_a^{(n+1)} - T_a^{(n)}}{\Delta t} = -2\varepsilon \sigma T_a^4
```
Since there is no source, the temperature will exponentially decrease until it reaches equilibrium at 0 K. \
let us define
```math
D = \frac{2 \varepsilon \sigma \left( T_a^{(n)}\right)^3}{C_a}
```

The update rule for the explicit time stepping is
```math
T_a^{(n+1)} = T_a^{(n)}(1 - D \Delta t)
```

We know that temperature should remain positive
```math
\frac{T_a^{(n+1)}}{T_a^{(n)}} = (1 - D\Delta t)> 0 \ .
```
This translates in the condition on ``\Delta t``
```math
\Delta t < \frac{1}{D} = \frac{C_a}{2\varepsilon \sigma \left( T_a^{(n)}\right)^3}
```
For ``T_a`` equal to 288 K, ``\Delta t`` should be lower than $(@sprintf "%.2f" Cₐ / (2 * 0.5 * σ * 288^3)) days \
Going back to the previous plot, which combination of parameters will make my model the most unstable?

What happens for implicit time stepping? We have that
```math
\frac{T_a^{(n+1)}}{T_a^n} = \frac{1}{1 + D\Delta t} > 0
```
Since ``D > 0``, implicit time stepping is stable for _any_ positive ``\Delta t`` \

In summary, for an ODE: 
- _Explicit time stepping_ is generally **_conditionally stable_**, i.e. the discrete system is stable given ``\Delta t < C`` where ``C`` depends on the system. 
- _Implicit time stepping_, on the other hand, is generally **_unconditionally stable_**
"""

# ╔═╡ ea517bbf-eb14-4d72-a4f4-9bb823e02f88
md"""
# Predicting earth's temperature distribution
"""

# ╔═╡ 140bcdac-4145-47b3-952f-bfe50f6ed41c
md"""
$(Resource("https://www.researchgate.net/profile/Anders-Levermann/publication/274494740/figure/fig9/AS:668865801506834@1536481442913/a-Surface-air-temperature-as-a-function-of-latitude-for-land-dashed-line-corrected.png", :height => 400))

**Figure**: Observed temperature profile from: Feulner et al, _On the Origin of the Surface Air Temperature Difference between the Hemispheres in Earth's Present-Day Climate_ (2013), Journal of Climate.
"""

# ╔═╡ 849775fa-4990-47d3-afe0-d0a049bb90af
md"""
We download the annually and zonally average observed temperature and radiation profiles from `https://github.com/simone-silvestri/ComputationalThinking/raw/main/` using the Julia package DataDeps and open it using the JLD2 package
"""

# ╔═╡ 4d517df8-0496-40a2-8e44-5beda6cd7226
# ╠═╡ show_logs = false
begin
	# We use the package DataDeps to download the data stored at `online_path`
	ENV["DATADEPS_ALWAYS_ACCEPT"]="true"	
	
	online_path = "https://github.com/simone-silvestri/ComputationalThinking/raw/main/"

	dh = DataDep("computional_thinking_data",
	    "Data for class",
	    [online_path * "observed_radiation.jld2",   # Observed ASR and OLR (absorbed shortwave and outgoing longwave radiation)
	     online_path * "observed_T.jld2"] 		    # Observed temperature
		)

	DataDeps.register(dh)

	datadep"computional_thinking_data"

	obs_temp_path = @datadep_str "computional_thinking_data/observed_T.jld2"
	obs_rad_path  = @datadep_str "computional_thinking_data/observed_radiation.jld2"
	
	# Load the observed zonally and yearly averaged temperature profile
	T_obs = jldopen(obs_temp_path)["T"][1:2:end]
end

# ╔═╡ 6932b969-0760-4f09-935a-478ac56de262
md""" ε $(@bind ε PlutoUI.Slider(0:0.01:1, show_value=true, default = 0.0)) """

# ╔═╡ f2510e0a-23f2-4a40-a7db-7b59898facfa
begin
	# variable albedo
	# α₀ = 0.2855
	# α₁ = 0.1433
	# variable_albedo = @. α₀ + α₁ .* ϕ^2

	# variable albedo
	α₀ = 0.312
	α₁ = 0.15 
	variable_albedo(model) = @. α₀ + α₁ .* 0.5 * (3 * sin(model.ϕᶜ)^2 .- 1)

	# variable emissivity (function that depends on the model state)
	ε₀, ε₁, ε₂ = (0.75, 0.02, 0.005)
	function variable_emissivity(model) 
		return @. clamp(ε₀ + ε₁ * log2(430.0/280) + ε₂ * (model.Tₛ - 286.38), 0, 1.0)
	end
end

# ╔═╡ 901548f8-a6c9-48f8-9c8f-887500081316
md"""
# Heat transport

We have seen that the latitudinal temperature gradient generates a global circulation that transports heat from the equator to the poles. It is too computationally expensive to solve the governing equations here (General Circulation Models, or GCMs run ono supercomputers for days to solve the climate system). So we have to model our latitudinal transport in an easier way
"""

# ╔═╡ 590c1026-8e82-4dc7-a07b-6f3b96fbc0ee
md"""
### Global atmospheric circulation

There are three main factors that we have to take into account when considering large scale atmospheric circulation:
- hot air rises
- cold air sinks
- Coriolis force pushes winds to the right in the upper hemisphere and to the left in the lower hemisphere

Hot air in the equator rises upwards and moves towards the pole. It cools down in the process and about 30ᵒ it starts to sink creating large circulation cells called Hadley cells. At the poles, cold dense air tends to sink and move down towards the equator, creating the Polar pressure cells. While moving toward the equator, the air coming from the pole encounters faster spinning latitudes and is, therefore, diverted by the Coriolis effect. In between these two major cells, we form strong winds which are diverted towards the east (westerlies)

$(Resource("https://tdgil.com/wp-content/uploads/2020/04/Hadley-Cells-and-Wind-Directions.jpg", :height => 500))
**Figure**: schematic depicting the global atmospheric circulation

Global circulation requires the solution of a complex system of partial differential equations on the sphere. These equations (named Navier-Stokes equations) the conservation of mass, momentum, and energy in the climate system.
```math
\begin{align}
 & \frac{\partial\boldsymbol{u}}{\partial t} + (\boldsymbol{u} \cdot \boldsymbol{\nabla})\boldsymbol{u} + f\widehat{\boldsymbol{z}} \times \boldsymbol{u} = - \frac{\boldsymbol{\nabla} p}{\rho} + \boldsymbol{\nabla}\Phi \\
& \frac{\partial T}{\partial t} +  \boldsymbol{u} \cdot \boldsymbol{\nabla} T = R_{in} - R_{out}(T^4) \\
& \frac{\partial \rho}{\partial t} + \boldsymbol{\nabla} \cdot (\rho \boldsymbol{u}) = 0 \\
\end{align}
```
complemented by an equation of state (usually ideal gas) in the form ``p = EOS(\rho, T)``. General Circulation Models (or GCMs) solve this system of equations on a discrete three-dimensional grid to provide velocities and temperatures on the surface and in the atmosphere.
"""

# ╔═╡ 567fa8d3-35b4-40d7-8404-ae78d2874380
md"""
## Modeling latitudinal transport
what do we have to add to our model to include atmospheric circulation?

```math
\begin{align}
C_a \frac{dT_a}{dt} & = \varepsilon \sigma T_s^4 - 2\varepsilon \sigma T_a^4 + \tau_a \\
C_s \frac{dT_s}{dt} & =  \varepsilon \sigma T_a^4 - \sigma T_s^4 + (1 - \alpha) Q + \tau_s
\end{align}
```

where ``\tau_a`` and ``\tau_s`` represent the source/sink caused by heat transported around by currents in the atmosphere and in the ocean.

How can we calculate these additional terms?

$(Resource("https://raw.githubusercontent.com/simone-silvestri/ComputationalThinking/main/schematic.png", :height => 400))
**Figure**: energy fluxes in our control volume.

In our control volume we have a flux in ``F^+`` and a flux out ``F^-``
The energy stored in the control volume (``\tau``) will be the difference of the fluxes divided by the surface are of the control volume, where
```math
A = \underbrace{2\pi R \cos{\phi}}_{\text{circumference}} \cdot \underbrace{R\Delta \phi}_{\text{width}}
```
So 
```math
\tau = - \frac{1}{2\pi R^2 \cos{\phi}} \frac{F^- - F^+}{\Delta \phi}
```
taking ``\Delta \phi \rightarrow 0``
```math
\tau = - \frac{1}{2\pi R^2 \cos{\phi}} \frac{\partial F}{\partial \phi}
```

How can we represent ``F``? 
Mathematically, the flux at the boundary of a computational element is calculated as ``(V\cdot T)``. The velocity ``V`` is the flow velocity at the interface of the element, which is determined by the Navier-Stokes equations, 3-dimensional PDEs that ensure momentum conservation in fluid dynamic systems. These equations are notoriously hard to solve, they require very fine grids due to the chaotic nature of fluid flows. Therefore, we will take a shortcut and _parametrize_ the flux at the interface (i.e., approximate the flux with a semi-empirical model).
We can think at the transport of heat by the atmosphere as moving heat from _HOT_ to _COLD_ regions of the earth. In general, this holds if we zoom out enough. We can think of the transport by the atmosphere as a diffusion process, which goes "DOWN" the gradient of temperature. As such we can parametrize the heat flux with an effective conductivity
```math
F \approx - 2\pi R^2 \cos{\phi} \cdot \kappa \frac{\partial T}{\partial \phi}
```
where ``\kappa`` is the "conductivity" of our climate system in W/(m²K) due to the movement in the atmosphere
And, finally assuming that ``\kappa`` does not vary in latitude (very strong assumption!) we can model the heat source due to transport as
```math
\tau = \frac{\kappa}{\cos{\phi}} \frac{\partial}{\partial \phi} \left(\cos{\phi}  \frac{\partial T}{\partial \phi} \right)
```

NOTE: in a metal rod, where the area does not vary with length, the ``cos`` terms drop and a heat diffusion can be modelled with just
```math
\tau \approx D \frac{\partial^2 T}{\partial x^2}
```
"""

# ╔═╡ 0d8fffdc-a9f5-4d82-84ec-0f27acc04c21
md"""
## Let's code it up!

Our governing system of equation is now a system of PDE, so we have to discretize the ``\phi``-direction as well a the time.

```math
\begin{align}
C_a \frac{\partial T_a}{\partial t} & = \sigma T_s^4 - 2\varepsilon \sigma T_a^4 + \frac{\kappa}{\cos{\phi}} \frac{\partial}{\partial \phi} \left(\cos{\phi}  \frac{\partial T_a}{\partial \phi} \right)\\
C_s \frac{\partial T_s}{\partial t} & = - \sigma T_s^4 + \varepsilon \sigma T_a^4 + (1 - \alpha) Q + \frac{\kappa}{\cos{\phi}} \frac{\partial}{\partial \phi} \left(\cos{\phi}  \frac{\partial T_s}{\partial \phi} \right)
\end{align}
```

We need to define a ``\Delta x`` (or ``\Delta \phi`` in our case) and we can discretize a spatial derivative in the same way as the time-derivative
```math
\left[\frac{1}{\cos{\phi}} \frac{\partial}{\partial \phi} \left(\cos{\phi}  \frac{\partial T}{\partial \phi} \right) \right]_j \approx \frac{1}{\cos{\phi_j} \Delta \phi} \left(\left[ \cos{\phi} \frac{\partial T}{\partial \phi} \right]_{j+1/2} - \left[ \cos{\phi} \frac{\partial T}{\partial \phi} \right]_{j-1/2} \right)
```
In the same way we can approximate the first derivative on interfaces as
```math
\left[\cos{\phi}\frac{\partial T}{\partial \phi} \right]_{j+1/2} \approx \cos{\phi_{j+1/2}}\frac{T_{j+1} - T_{j}}{\Delta \phi}
```
The additional tendency term caused by heat transport becomes:
```math
G_\kappa = \frac{\kappa}{\cos{\phi_j}\Delta \phi} \left(\cos{\phi_{j+1/2}} \frac{T_{j+1} - T_j}{\Delta \phi} - \cos{\phi_{j-1/2}} \frac{T_{j} - T_{j-1}}{\Delta \phi} \right)
```
"""


# ╔═╡ abdbbcaa-3a76-4a47-824d-6da73bc71c31
Model(; κ = 0.25)

# ╔═╡ 1cef338d-5c4a-4ea5-98d7-9ac4f11922f3
md"""
#### Coding explicit time stepping with diffusion

Coding the explicit time stepping won't be too different than what we did with the ODE. \ 
We define the tendencies and add the explicit diffusion term calculated as above. \
This time we need boundary conditions for our PDE. 

We assume that there are no fluxes over the poles.
```math
F^+(90^\circ) = F^-(-90^\circ) = 0
```

"""

# ╔═╡ 71cff056-a36c-4fd4-babb-53018894ac5c
begin
	function explicit_diffusion(T, Δϕ, ϕᶠ, ϕᶜ)
		# Calculate the flux at the interfaces
		Flux = cos.(ϕᶠ[2:end-1]) .* (T[2:end] .- T[1:end-1]) ./ Δϕ
		# add boundary conditions
		# We impose 0-flux boundary conditions
		Flux = [0.0, Flux..., 0.0]
		return 1 ./ cos.(ϕᶜ) .* (Flux[2:end] .- Flux[1:end-1]) ./ Δϕ
	end
	
	function tendencies(model)
		Tₛ = model.Tₛ
		Tₐ = model.Tₐ
		α  = albedo(model)
		ε  = emissivity(model)

		Δϕ = model.ϕᶠ[2] - model.ϕᶠ[1]
		Dₛ = diffusivity(model) .* explicit_diffusion(model.Tₛ, Δϕ, model.ϕᶠ, model.ϕᶜ)
		Dₐ = diffusivity(model) .* explicit_diffusion(model.Tₐ, Δϕ, model.ϕᶠ, model.ϕᶜ)

		Gₛ = @. (1 - α) * model.Q + σ * (ε * Tₐ^4 - Tₛ^4) + Dₛ
		Gₐ = @. σ * ε * (Tₛ^4 - 2 * Tₐ^4) + Dₐ
		return Gₛ, Gₐ
	end
end

# ╔═╡ a93c36c9-b687-44b9-b0b6-5fe636ab061c
function time_step!(model::ExplicitModel, Δt)
	# Calculate rhs of the two equation (also called tendencies)
	Gₛ, Gₐ = tendencies(model)

	# update temperature to time step n+1
	@. model.Tₛ += Δt * Gₛ / model.Cₛ
	@. model.Tₐ += Δt * Gₐ / model.Cₐ
end

# ╔═╡ 0ed9c5d9-8bef-4c1e-9be3-5aff1a7d226c
function evolve_model!(model; Δt = 30.0, stop_year = 40)
	stop_iteration = ceil(Int, stop_year * 365 ÷ Δt)
	for iter in 1:stop_iteration
		time_step!(model, Δt)
	end
end

# ╔═╡ 1d8a69b7-52db-4865-8bf2-712c2b6442f5
# ╠═╡ show_logs = false
begin 
	# construct and evolve a radiative model with constant ε
	model_lat = Model(; ε)

	ϕ = model_lat.ϕᶜ .* 180 / π
	
	evolve_model!(model_lat, Δt = 80, stop_year = 800);

	# plot the latitudinal dependent temperatures
	plot_latitudinal_variables!(ϕ, [T_obs.-273.15, model_lat.Tₛ .- 273.15];
								labels = ["observed T", "modelled T"],
								colors = [:black, :red], 
								styles = [:dashdot, :solid],
								ylims = (-60, 50), 
								title = "emissivity: $ε");
	md""" 
	$(current_figure())
	**Figure**: Comparison between observed (dashed-dotted line) and temperature calculated by the model
	"""
end

# ╔═╡ 5884999f-f136-4ae7-8831-cc3a36f50a98
begin
	ASR_obs = jldopen(obs_rad_path)["ASR"][1:2:end]
	ASR(model) = (1 .- albedo(model)) .* model.Q 
	
	plot_latitudinal_variables!(ϕ, [ASR(model_lat), ASR_obs];
								labels = ["modeled ASR [W/m²]", "observed ASR [W/m²]"],
								colors = [:red, :red], 
								styles = [:solid, :dashdot],
								ylabel = "")
	md""" 
	## Improving the model

	the model we constructed has several approximations which we can improve on
	
	### Latitude-dependent albedo

	Till now, we assumed the albedo to be equal to 0.2985 because it gave us the best fit with observations in an averaged global scenario. However, is a constant albedo a good approximation when considering latitudinal dependency? We can infer the earth's _real_ albedo by looking at the absorbed radiation at the surface (ASR or absorbed shortwave radiation). In our model we can calculate it as such
	```
	ASR(model) = @. (1 - albedo(model)) * model.Q
	```
	$(current_figure())
	**Figure**: comparison between ASR used to force the RadiativeModel and the observed ASR (from [NCEP reanalysis](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html))

	The ASR seems lower at the poles, suggesting that the albedo is higher in those regions. This is a result of the lower sun angle present at the poles but also, the higher presence of fresh snow, ice, and smooth open water- all areas prone to high levels of reflectivity
	
	``\alpha`` can be approximated with a function of ``\sin{\phi}`` that allows us to have a lower albedo at the poles than the one at the equator
	```math
	\alpha(\phi) = \alpha_0 + \alpha_1 (3 * sin(\phi)^2 - 1)
	```
	### Temperature-dependent emissivity

	If the temperature rises, the saturation pressure rises, allowing more water vapor to remain in equilibrium in the gaseous form in the atmosphere. Since the emissivity is largely dependent on the water vapor content, a temperature rise causes an increase in the atmosphere's emissivity. You have seen in 2.2 that this _feedback_ effect can be modeled with a linear function of temperature:
	```math
	\varepsilon = \varepsilon_0 + \varepsilon_1 \log{\frac{\text{CO}_2}{{\text{CO}_2}_{\text{PRE}}}} + \varepsilon_2(T-T_{\text{PRE}})
	```
	Where the subscript ``\text{PRE}`` indicates pre-industrial values. Remember! The temperature in the above formula is the _surface_ temperature. The effect of water vapor is not hugely important in the climate change context, as the vapor pressure of H₂O is capped by the saturation pressure, but is of great importance in shaping the latitudinal temperature profile (it is much colder in the poles that in the equator).
	To allow a temperature-dependent emissivity, we have to extend the ```emissivity``` method to ensure it can accept functions. We can define a function that accepts the model as an input and returns the temperature-dependent emissivity and use it as an input to our model
	```
	varε(model) = ε₀ + ε₁ * log2(440.0/280) + ε₂ * (model.Tₛ - 286.38)
	```
	We that have to dispatch the emissivity function to behave in a different way when the ε field of our RadiativeModel is a function	
	```
	emissivity(model::RadiativeModel{<:Any, <:Any, <:Function}) = model.ε(model)
	```
	Note! Physical values of emissivity range between 0 and 1!
	"""
end

# ╔═╡ 4640a179-3373-4901-ac31-31022e8c7eb2
begin
	reference_model = Model(; ε = 0.8)
	feedback_model  = Model(; ε = variable_emissivity, α = variable_albedo)

	evolve_model!(reference_model, Δt = 50, stop_year = 1200)
	evolve_model!(feedback_model,  Δt = 50, stop_year = 1200)

	α_obs = 1 .- ASR_obs ./ feedback_model.Q

	fig_temp = plot_latitudinal_variables!(ϕ, [albedo(feedback_model), ϕ ./ ϕ .* 0.2985, α_obs];
										   labels = ["varα(model)", "α = 0.2985", "observed α"], 
										   colors = [:blue, :red, :black], 
										   styles = [:solid, :solid, :dashdot], 
										   ylabel = "albedo",
										   ylims = (0.0, 0.8), 
										   leg_pos = :ct, 
										   res = (700, 600))
	
	fig_temp = plot_latitudinal_variables!(ϕ, [emissivity(feedback_model), ϕ ./ ϕ .* 0.75];
										   labels = ["varε(model)", "ε = 0.75"], 
										   colors = [:blue, :red], 
										   styles = [:solid, :dash], 
	                                       ylabel = "emissivity",
										   fig = fig_temp, 
										   ax_pos = [1, 2],
										   ylims = (0.2, 1.0))
	
	fig_temp = plot_latitudinal_variables!(ϕ, [feedback_model.Tₛ .- 273.15,
											   reference_model.Tₛ .- 273.15,
											   T_obs .- 273];
										   labels = ["variable ε and α",
											   "constant ε and α", 
											   "observed T"], 
										   colors = [:blue, :red, :black, :purple], 
										   styles = [:solid, :solid, :dashdot, :dash], 
										   ylims = (-90, 70),
										   ax_pos = [2:2, 1:2], 
										   fig = fig_temp)
	
	md"""
	Let's take a look at our final latitudinal temperature model, complete with varying emissivity and albedo. 

	$(current_figure())
	**Figure**: comparison between observed temperature (dashed-dotted line), temperature calculated from a model with constant ``\varepsilon`` and ``\alpha`` (red) and from a model with latitude-dependent ``\alpha`` and water vapor feedback (blue)

	The prediction worsens when compared to the simple constant emissivity/constant albedo model. This is usually a sign that we are neglecting some important physical phenomenon.
	"""
end

# ╔═╡ d13c319d-345a-40b8-b90d-b0b226225434
begin
	OLR_obs = jldopen(obs_rad_path)["OLR"][1:2:end]

	OLR(model) = σ .* ((1 .- emissivity(model)) .* model.Tₛ.^4 + emissivity(model) .* model.Tₐ.^4)

	plot_latitudinal_variables!(ϕ, [ASR_obs, ASR(feedback_model), OLR_obs, OLR(feedback_model)];
								labels = ["Observed ASR",
								 		  "Modeled ASR",
								 		  "Observed OLR",
								 		  "Modeled OLR"],
								colors = [:red, :red, :blue, :blue],
								styles = [:dashdot, :solid, :dashdot, :solid],
								ylabel = "ASR and OLR [W/m²]")
	
	md"""
	### Outgoing Longwave Radiation (OLR)

	Outgoing longwave radiation is the energy that the earth loses to space in the form of radiative emission. It has a contribution from the atmosphere and a contribution from the emitted energy from the surface that manages to escape from the absorption of the atmosphere
	```math
	OLR = (1-\varepsilon) \sigma T_s^4 + \varepsilon \sigma T_a^4
	```
	in our model
	```
	OLR(model) = @. (1 - emissivity(model)) * σ * model.Tₛ^4 + emissivity(model) * σ * model.Tₐ^4
	```

	$(current_figure())
	**Figure**: comparison between observed and calculated ASR and OLR

	The OLR and ASR from our model match quite closely. This is expected: since every latitude in our model is independent, the incoming energy (ASR) must match the outgoing energy (OLR) for energy conservation to hold. This is not the case for the observed profiles. We see that at the equator the incoming energy is larger than the emission and vice-versa happens at the poles. This is an indication that, in the real climate system, energy is transported from the equator to the poles. The mechanism that allows this heat transport is the presence of a global atmospheric circulation. 
	"""
end

# ╔═╡ 83be4f9d-6c85-4e5b-9379-00618c9e39be
md"""
#### Coding implicit time stepping with diffusion

to code an implicit time stepping method, we can reutilize the matrix we used before (the sources and interexchange terms do not change)
There are new terms to be added:

```math
G_\kappa^{(n+1)} = \frac{\kappa}{\cos{\phi_j}\Delta \phi} \left(\cos{\phi_{j+1/2}} \frac{T_{j+1}^{(n+1)} - T_j^{(n+1)}}{\Delta \phi} - \cos{\phi_{j-1/2}} \frac{T_{j}^{(n+1)} - T_{j-1}^{(n+1)}}{\Delta \phi} \right)
```
we can rearrange it as
```math
G_\kappa^{(n+1)} = \kappa\frac{\cos{\phi_{j-1/2}}}{\cos{\phi_j}\Delta\phi^2}  T_{j-1}^{(n+1)} + \kappa\frac{\cos{\phi_{j+1/2}}}{\cos{\phi_j}\Delta\phi^2}  T_{j+1}^{(n+1)} - \kappa\frac{\cos{\phi_{j+1/2}} + \cos{\phi_{j-1/2}}}{\cos{\phi_j}\Delta\phi^2} T_j^{(n+1)}
```
```math
G_\kappa^{(n+1)} = a T_{j-1}^{(n+1)} + c T_{j+1}^{(n+1)} - (a+c) T_j^{(n+1)}
```
where 
```math
a = \kappa\frac{\cos{\phi_{j-1/2}}}{\cos{\phi_j}\Delta\phi^2} \ \ \ \text{and} \ \ \ c = \kappa\frac{\cos{\phi_{j+1/2}}}{\cos{\phi_j}\Delta\phi^2}
```
we have to add ``(a+c)`` to the diagonal, ``a`` to the diagonal at position ``-1`` and ``c`` to the diagonal at position ``+1``

**_Adding boundary conditions_** \
at ``j = 1/2`` we impose a no flux boundary condition. This implies that
```math
F_{1/2} = 0 \Rightarrow T_0 = T_1
```
then ``G_{\kappa}^{(n+1)}`` at 1 simplifies to
```math
\left[G_\kappa^{(n+1)}\right]_1 = c T_{2}^{(n+1)} - c T_1^{(n+1)}
```
Simply put, we avoid adding ``a`` to the first row. The same happens for ``G_{\kappa}^{(n+1)}`` at ``m`` (with ``m`` the length of ``\phi``) where, assuming that ``F_{m+1/2} = 0`` implies that
```math
\left[G_\kappa^{(n+1)}\right]_m = a T_{m-1}^{(n+1)} - a T_m^{(n+1)}
```
"""

# ╔═╡ 7c7439f0-d678-4b68-a5e5-bee650fa17e2
function construct_matrix(model, Δt)

	A = construct_radiative_matrix(model, Δt)
	
	cosϕᶜ = cos.(model.ϕᶜ)
	Δϕ = model.ϕᶠ[2] - model.ϕᶠ[1]

	κ = diffusivity(model)
	
	a = @. κ / Δϕ^2 / cosϕᶜ * cos(model.ϕᶠ[1:end-1])
	c = @. κ / Δϕ^2 / cosϕᶜ * cos(model.ϕᶠ[2:end])

	m = length(model.Tₛ)
    for i in 1:m
		# Adding the off-diagonal entries corresponding to Tⱼ₊₁ (exclude the last row)
        if i < m
            A[i  , i+1]   = - Δt * c[i]
            A[i+m, i+1+m] = - Δt * c[i]
		end
		# Adding the off-diagonal entries corresponding to Tⱼ₋₁ (exclude the first row)
        if i > 1 
            A[i,   i-1]   = - Δt * a[i]
            A[i+m, i-1+m] = - Δt * a[i]
        end
		# Adding the diagonal entries
        A[i  , i]   += Δt * (a[i] + c[i])
        A[i+m, i+m] += Δt * (a[i] + c[i])
    end
	
	return A
end

# ╔═╡ d1a741ad-f28d-48c7-9f15-d55d0801573d
md"""
## Temperature distribution with diffusion
"""

# ╔═╡ a046b625-b046-4ca0-adde-be5249a420f4
md""" κ $(@bind κ_slider PlutoUI.Slider(0:0.1:1, show_value=true)) """

# ╔═╡ 514ee86b-0aeb-42cd-b4cd-a795ed23b3de
begin
	diffusive_model = Model(; κ = κ_slider, ε = variable_emissivity, α = variable_albedo)
	
	evolve_model!(diffusive_model, Δt = 50, stop_year = 1200)
	
	DTκ0 =  feedback_model.Tₛ[22] -  feedback_model.Tₛ[end-6]
	DTκ1 = diffusive_model.Tₛ[22] - diffusive_model.Tₛ[end-6]
	DTob = T_obs[22] - T_obs[end-6]

	title = @sprintf("T(0ᵒ) - T(90ᵒ): %.2f (κ = 0), %.2f (κ = %.2f), %.2f (obs)", 
					 DTκ0, DTκ1, κ_slider, DTob)
	
	fig_diff = plot_latitudinal_variables!(ϕ, [ feedback_model.Tₛ .- 273, 
									diffusive_model.Tₛ .- 273,
									T_obs .- 273];
								labels = ["model with κ = 0",
									 "model with κ = $κ_slider",
									 "observed T"], 
								colors = [:blue, :blue, :black], 
								styles = [:dash, :solid, :dashdot],
								ylims = (-90, 70), title)
	
	HF(model) = model.κ .* explicit_diffusion(model.Tₛ, deg2rad(2), model.ϕᶠ, model.ϕᶜ) 

	plot_latitudinal_variables!(ϕ, [ASR(diffusive_model), 
									OLR(diffusive_model), 
									HF(diffusive_model),
									ASR_obs,
									OLR_obs,
									OLR_obs .- ASR_obs], 
								labels = ["ASR", "OLR", "transport"], 
								colors = [:red, :blue, :green, :red, :blue, :green], 
								styles = [:solid, :solid, :solid, :dash, :dash, :dash],
								ylabel = "Energy Budget Wm⁻²",
								leg_pos = :cc,
								ax_pos = (1, 2),
								fig = fig_diff)
	current_figure()	
end

# ╔═╡ c98fcf26-d715-47c2-84ac-63bffe02d813
md"""
Is this a sensible conductivity value? Heat transport is caused by global circulation, so what is a plausible value of the diffusivity caused by the oceanic motion? If we take into account the velocity of a typical ocean eddy ``V``, the order of magnitude of the flux is ``V \cdot T`` (where ``T`` is a temperature scale). We parameterize this with a term that looks like ``K \cdot T / L`` where ``K`` is our diffusivity expressed in m²/s and ``L`` is a typical length scale of oceanic motion. Then it should hold that
```math
K \cdot\frac{T}{L} \sim V \cdot T 
```
If we fill in typical values for ocean's velocity and length scales (about ``0.5`` m/s and ``100`` km) we get that
```math
K \sim V\cdot L \sim 5 \cdot 10^4 \ \ \text{m}^2/s
```
This diffusivity corresponds to a conductivity ``\kappa`` (in W/m²K) of
```math
\kappa \sim \frac{K C_s}{R^2} \approx 0.49 \ \ \text{W}/\text{m}^2\text{K}
```
"""

# ╔═╡ 77a73a9d-9d78-4cf5-ae19-f1107fa5b4c2
begin
	md"""
	
	Heat flux can only redistribute energy. It is then important to check that our solution method does not create or destroy energy.
	Let's compute the integral of the transport term in latitude
	```math
	\int_{-90^o}^{90^o} \tau \cos{\phi} d\phi \approx \sum_{j = 1}^N \tau_j \cos{\phi_j} \cdot (\phi_{j+1/2} - \phi_{j-1/2}) 
	```
	This is a good sanity check to make sure that our model is doing what is intended
	"""
end

# ╔═╡ 80c72898-139e-44af-bab0-ca638f282188
sum(HF(diffusive_model) .* cos.(diffusive_model.ϕᶜ) .* (diffusive_model.ϕᶠ[2:end] .- diffusive_model.ϕᶠ[1:end-1]))

# ╔═╡ b0ca64b8-0211-4d1c-b007-7583bf8ac908
md"""
#### Stability of a diffusive model

Let's, once again, reduce the two equations to a more simple, 1D partial differential equation, which only has a diffusion term. You can imagine that greenhouse gases are immediately removed from the atmosphere. As a result, the atmosphere stops absorbing heat from the surface and emitting it in space. The remaining heat then will only redistribute via transport along the atmosphere. (we also simplify the earth to be "flat", i.e., no cosines)

```math
\frac{\partial T}{\partial t} = K \frac{\partial^2 T}{\partial x^2}
```
where ``K`` is the diffusivity in m²/s
```math
\frac{T^{(n+1)}_j - T^{(n)}_j}{\Delta t} = K \frac{T^{(n)}_{j+1} - 2T^{(n)}_j+ T^{(n)}_j}{\Delta x^2}
```

Imagine the initial temperature profile can be approximated by a spatial wave of wavenumber ``k``, i.e,
```math
T^{(n)}_j = \xi^{(n)}e^{ikx_j} \ , \ \ \ \ \text{where} \ \ \ \ x_j = j\Delta x
```

Let's substitute in the following equation and divide by ``\xi^{(n)}e^{ikj\Delta x}``
```math
\left( \frac{\xi^{(n+1)}}{\xi^{(n)}} - 1\right) = K\Delta t \frac{e^{ik \Delta x} - 2 + e^{- ik \Delta x}}{\Delta x^2}
```

we can use ``e^{i\theta} + e^{-i\theta} = 2\cos{\theta}`` and we get
```math
\frac{\xi^{(n+1)}}{\xi^{(n)}} = 1 +\frac{K \Delta t}{\Delta x^2} \underbrace{\left( 2\cos{k\Delta x} - 2\right)}_{\text{between \ } -4 \text{ \ and \ } 0}
```

The worst-case scenario occurs for wavenumbers ``k`` which give 
```math
\frac{\xi^{(n+1)}}{\xi^{(n)}} = 1 - 4\frac{K \Delta t}{\Delta x^2} 
```
again, we want the amplitude to remain positively correlated (otherwise it means that heat transfers from cold to hot temperatures), so we must ensure that 
```math
\Delta t < \frac{\Delta x^2}{4K}
```
In the previous case, we had that at temperatures which were reasonable for the atmosphere, the limitation was in the tenth of days... \
We already saw that a reasonable diffusivity (in m²/s) for the ocean is 
```math
K \approx 5 \cdot 10^4 \ \ \text{m}^2/\text{s}
```
atmospheric diffusivity is a higher because (1) heat capacity is lower at the same value of thermal conductivity or (2) velocity is higher and length scales are larger than the ocean. We can estimate that for the atmosphere the velocity and length scales are roughly in the order of 10 ms⁻¹ and 1000 km, respectively. This gives us a diffusivity
```math
K \sim V\cdot L \approx 10 \cdot 10^6 = 10^7 \ \ \ \text{m}^2\text{/s}
```
If we have a model with a two-degree resolution (90 points), then ``\Delta x = R \Delta \phi \approx 200`` km, which means that the condition on the time step is
```math
\Delta t < \frac{(2\cdot 10^{5})^2}{1 \cdot 10^7}  \approx 1 \text{hr} !
```
We introduced another limitation in our explicit time stepping. The time step is connected with the spatial resolution of our model. This limitation is called CFL condition (Courant–Friedrichs–Lewy condition) and ensures that the temperature does not move more than one spatial grid cell in one-time step. 

**do it by yourself** \
Demonstrate that implicit time stepping does not have the same limitations
"""

# ╔═╡ 8767e475-e842-43d6-8797-8bcbaa51b5cd
md"""
# Parameter Estimation

Write here about parameter estimation!
"""

# ╔═╡ eee25676-16d0-4479-8887-e3fe6ffe5420
# Code credit to Andre Souza
function optimize!(θ, forward_map, observed_data; Niter = 10, Δt = 1.0)
    
    Ndata 	  = length(observed_data)
    Nensemble = length(θ)
    
    θseries = [copy(θ)]

	eval  = forward_map(mean(θ))

    error = norm(eval - observed_data)
    @info "iteration 0 with error $error"

	G = [copy(eval) for i in 1:Nensemble]

    @inbounds for i = 1:Niter
        θ̄ = mean(θ)
        Threads.@threads for n in 1:Nensemble
			G[n] .= forward_map(θ[n]) # error handling needs to go here
		end
		G̅ = mean(G)

        # define covariances
        Cᵘᵖ = (θ[1] - θ̄) * (G[1] - G̅)'
        Cᵖᵖ = (G[1] - G̅) * (G[1] - G̅)'
        for j = 2:Nensemble
            Cᵘᵖ += (θ[j] - θ̄) * (G[j] - G̅)'
            Cᵖᵖ += (G[j] - G̅) * (G[j] - G̅)'
        end
        Cᵘᵖ *= 1 / (Nensemble - 1)
        Cᵖᵖ *= 1 / (Nensemble - 1)

        # ensemblize the data
        y = [observed_data + Δt * randn(Ndata) for i = 1:Nensemble]
        r = y - G

        # update
        Cᵖᵖ_factorized = cholesky(Symmetric(Cᵖᵖ + 1 / Δt * LinearAlgebra.I))
        for j = 1:Nensemble
            θ[j] .+= Cᵘᵖ * (Cᵖᵖ_factorized \ r[j])
        end
    
        error = norm(forward_map(mean(θ)) - observed_data)
        @info "iteration $i with error $error"
        push!(θseries, copy(θ))
    end

    return θ, θseries
end

# ╔═╡ bace5438-1d2a-45aa-b3f0-1780db5857be
function forward_map(θ)
	κ = abs(θ[1])
	model = Model(; κ, ε = variable_emissivity, α = variable_albedo)

	model.Tₛ .= diffusive_model.Tₛ
	model.Tₐ .= diffusive_model.Tₐ
	
	evolve_model!(model, Δt = 50, stop_year = 500)
	 
	return model.Tₛ[4:end-3]
end

# ╔═╡ 4384cc1d-348d-49a8-99c7-8c304e1e5d49
begin 
	θ = [0.2 .* rand(1) for i in 1:10]
	initial_θ = deepcopy(mean(θ))
	optimize!(θ, forward_map,  T_obs[4:end-3]; Niter = 10)
end

# ╔═╡ c64b1647-eece-47a8-b4c7-485e0a730507
begin
	params = mean(θ)

	optimized_model = Model(; κ = abs(params[1]), ε = variable_emissivity, α = variable_albedo)
	non_optimized_model = Model(; κ = abs(initial_θ[1]), ε = variable_emissivity, α = variable_albedo)

	evolve_model!(optimized_model, Δt = 50, stop_year = 1200)
	evolve_model!(non_optimized_model, Δt = 50, stop_year = 1200)

	fig2 = plot_latitudinal_variables!(ϕ, [T_obs, non_optimized_model.Tₛ, optimized_model.Tₛ];
		                              labels = ["observation", "non-optimized", "optimized"],
		                              colors = [:grey4, :coral1, :blue],
							 		  widths = [4, 2, 2],
									  title = @sprintf("Diffusivity: %.3f", params[1]))

	fig2 = plot_latitudinal_variables!(ϕ, [OLR_obs, OLR(non_optimized_model), OLR(optimized_model)];
		                              labels = ["observation", "non-optimized", "optimized"],
		                              colors = [:gray1, :coral1, :blue],
							 		  widths = [4, 2, 2],
									  ax_pos = [1, 2],
									  fig = fig2,
									  ylabel = "Outgoing Longwave radiation [W/m²]",
									  title = @sprintf("Diffusivity: %.3f", params[1]))
end

# ╔═╡ f720550a-6637-41ca-aaaa-8b7dd218a00f
md"""
# Using a dynamical core: SpeedyWeather
"""

# ╔═╡ 2d8ff967-4ed3-4cf5-a99e-786c5249ea51


# ╔═╡ 984fe3c6-b1a6-47be-8644-46ea44519286
spectral_grid = SpectralGrid(trunc = 63, nlev = 2)

# ╔═╡ 54622eca-6ae0-4580-a5d2-3997e2f57907
begin
	output = OutputWriter(spectral_grid, PrimitiveDry; output_vars = [:vor, :div, :temp, :u, :v])
	initial_model = PrimitiveDryModel(; spectral_grid, output)
end

# ╔═╡ 54c03392-bf2a-4fd1-9550-287fec6c3a80
begin
	initial_simulation = initialize!(initial_model)
	run!(initial_simulation, n_days = 300, output = true)
end

# ╔═╡ df3a3777-5660-4785-aea6-836442329dc2
begin 
	ds = NCDataset("run_0001/output.nc")
	Nt = length(ds["temp"][1, 1, 1, :])
	Tm = mean(ds["temp"][:, :, :, Nt-10:Nt], dims = (1, 4))[1, :, :, 1] .+ 273.15

	figs = Figure(resolution = (700, 300))
	axs  = Axis(figs[1, 1])
	lines!(axs, ϕ, T_obs)
	lines!(axs, ds["lat"][:], Tm[:, 2])
	lines!(axs, ϕ, optimized_model.Tₛ)

	current_figure()
end

# ╔═╡ 50b6ac84-0a73-4f04-8516-ecee25b6bfcb
md""" iterations $(@bind iter PlutoUI.Slider(1:Nt, show_value=true)) """

# ╔═╡ f010953a-eb94-4a7a-bd03-7c6a08cfced4
begin 
	u = reverse(ds["u"][:, :, 2, iter], dims = 2)
	v = reverse(ds["v"][:, :, 2, iter], dims = 2)
	T = reverse(ds["temp"][:, :, 2, iter], dims = 2)
	
	figdyn = Figure(resolution = (1200, 300))
	axdyn  = Axis(figdyn[1, 1])
	heatmap!(axdyn, u, colormap = :bwr)
	axdyn  = Axis(figdyn[1, 2])
	heatmap!(axdyn, v, colormap = :grays)
	axdyn  = Axis(figdyn[1, 3])
	heatmap!(axdyn, T, colormap = :magma)

	current_figure()
end

# ╔═╡ 67e9bad7-6b16-450c-84d8-9b9a13cd423a
begin	
	auxiliary_model = Model(; N = spectral_grid.nlat_half * 2, ε = variable_emissivity, α = variable_albedo)
	evolve_model!(auxiliary_model, Δt = 100, stop_year = 800)

	auxiliary_model.Tₛ
	temperature_relaxation = SpeedyWeather.MyTemperatureRelaxation(spectral_grid; 
															  		   temp_equi_a = auxiliary_model.Tₐ,
															  		   temp_equi_s = auxiliary_model.Tₛ)

	forced_model      = PrimitiveDryModel(spectral_grid; output, temperature_relaxation)
	forced_simulation = initialize!(forced_model)
	run!(forced_simulation, n_days = 300, output = true)
end
	

# ╔═╡ 51f3fd00-508b-4b42-bd95-ae19cb19b4db
# md"""
# ## Ice-albedo feedback and the Snowball earth

# The albedo of Earth's surface varies from about 0.1 for the oceans to 0.6–0.9 for ice and clouds — meaning that clouds, snow, and ice are good radiation reflectors while liquid water is not. 
# We can build this process in our model by imagining the earth covered by ice when we lower the temperature below a certain threshold ``T_{ice}``

# ```math
# \alpha(\phi, T) = \begin{cases} \alpha(\phi) & \ \ \text{if} \ \ T > T_{ice} \\ \alpha_{ice} &  \ \ \text{if} \ \ T \le T_{ice} \end{cases} 
# ```
# where ``\alpha(\phi)`` is our previously defined array `variable_albedo`, ``T_{ice} = -10`` ``^\circ``C and ``\alpha_{ice} = 0.7``
# """

# ╔═╡ 8afe64e3-d19a-4801-b7b8-56d886f7a59a
# plot_latitudinal_variables!(ϕ, [ASR(diffusive_model), 
# 								OLR(diffusive_model), 
# 								HF(diffusive_model)];
# 							labels = ["ASR", "OLR", "transport"], 
# 							colors = [:red, :blue, :green], 
# 							styles = [:solid, :solid, :solid],
# 							ylabel = "Energy Budget Wm⁻²",
# 							leg_pos = :cc)

# ╔═╡ ebcf224f-c006-4098-abf0-5c3644e2ee97
# md"""
# There will be a latitude where ``T_{\phi} = T_{ice}`` above which the earth will be covered in ice. We call this latitude ice line.
# Let's define an **ice_line** function that retreives this latitude
# """

# ╔═╡ 73238f6c-b8d3-4f92-bdfe-1c657e239903
# function ice_line(model)
# 	idx = searchsortedlast(model.Tₛ, 273.15 - 10)

# 	return idx == 0 ? 90.0 : idx > length(model.ϕᶜ) ? 0.0 :  - rad2deg(model.ϕᶜ[idx])
# end

# ╔═╡ 247b4c3a-2777-47ed-8cc9-e90a5cdb640b
# "the ice line of our model is at $(ice_line(diffusive_model)) ᵒN"

# ╔═╡ 0353c906-55d9-4419-a55d-8fcd374004d7
# 	function calc_different_climates(initial_condition_model; forcing)
# 		ice_line_model = zeros(length(forcing))
# 		for (idx, S₀) in enumerate(forcing)
# 			F(ϕ) = annual_mean_insolation.(ϕ; S₀)
# 			model_tmp = Model(; κ, ε = varε, α = α_model, Q = F)
	
# 			model_tmp.Tₛ .= initial_condition_model.Tₛ
# 			model_tmp.Tₐ .= initial_condition_model.Tₐ
	
# 			evolve_model!(model_tmp, Δt = 100, stop_year = 100)
	
# 			ice_line_model[idx] = ice_line(model_tmp)
# 		end
# 		return ice_line_model
# 	end

# ╔═╡ 1c33dc21-04af-4139-9061-696db73c3249
# begin 
# 	S₀₁ = range(1200.0, 1450, length = 25)
# 	ice_line_current = calc_different_climates(current_climate_model, forcing = S₀₁)

# 	figure_ice = Figure(resolution = (500, 300))
	
# 	ax_ice = Axis(figure_ice[1, 1], title = "ice line", ylabel = "ϕ [ᵒN]", xlabel = "forcing S₀ [W/m²]")
# 	lines!(ax_ice, S₀₁, ice_line_current, label = "initial condition: current climate")
# 	current_figure()
# end

# ╔═╡ 70713834-3246-45a4-a4c8-68513bb853ce
# md"""
# Let's start from another initial condition

# """

# ╔═╡ 2b33a8a1-3772-4fb3-a914-a20a7aae91bc
# begin
	# low_F = annual_mean_insolation.(ϕ; S₀ = S₀₁[1])
	# cold_climate_model = DiffusiveModel(ImplicitTimeStep(), length(low_F); κ, ε = varε, α = α_model, Q = low_F)
	# evolve_model!(cold_climate_model, Δt = 100, stop_year = 100)

	# new_current_climate_model = DiffusiveModel(ImplicitTimeStep(), length(low_F); κ, ε = varε, α = α_model, Q = F)
	# new_current_climate_model.Tₛ .= cold_climate_model.Tₛ
	# new_current_climate_model.Tₐ .= cold_climate_model.Tₐ
	
	# evolve_model!(new_current_climate_model, Δt = 100, stop_year = 1000)
	
	# plot_latitudinal_variables!(ϕ, [current_climate_model.Tₛ .- 273,
	# 								cold_climate_model.Tₛ .- 273, 
	# 								new_current_climate_model.Tₛ .- 273], 
	# 								["current climate",
	# 								 "cold climate (S₀ = 1200)",
	# 								 "current climate, different initial conditions"], 
	# 								[:blue, :blue, :red, :black], 
	# 								[:dash, :solid, :solid, :dashdot];
	# 								ylims = (-100, 70),
	# 								leg_pos = :lt)
# end

# ╔═╡ f7a96c67-bb62-4440-8c8a-f31ebe0aab7e
# md"""
# The solution is different! This is an example of a system depending on it's history (also called _hysteresis_). 
# """

# ╔═╡ b768707a-5077-4662-bcd1-6d38b3e4f929
html"""
        <style>
                main {
                        margin: 0 auto;
                        max-width: 2000px;
                padding-left: max(320px, 10%);
                padding-right: max(320px, 10%);
                }
        </style>
        """


# ╔═╡ 419c8c31-6408-489a-a50e-af712cf20b7e
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataDeps = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NCDatasets = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
SpeedyWeather = "9e226e20-d153-4fed-8a5b-493def4f21a9"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CairoMakie = "~0.10.12"
DataDeps = "~0.7.11"
JLD2 = "~0.4.38"
NCDatasets = "~0.13.1"
PlutoUI = "~0.7.53"
SpeedyWeather = "~0.6.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "156b46a97664e84f38f57642dc69608de27be70f"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "f35684b7349da49fcc8a9e520e30e45dbb077166"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "16267cf279190ca7c1b30d020758ced95db89cd0"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.5.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssociatedLegendrePolynomials]]
git-tree-sha1 = "3204d769e06c5678b23cf928d850f2f4ad5ec8a5"
uuid = "2119f1ac-fb78-50f5-8cc0-dda848ebdb19"
version = "1.0.1"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Automa]]
deps = ["TranscodingStreams"]
git-tree-sha1 = "ef9997b3d5547c48b41c7bd8899e812a917b409d"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.4"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitInformation]]
deps = ["Distributions", "Random", "StatsBase"]
git-tree-sha1 = "8f98d9d01f50d3a9bf987d7e206c993b390a98bf"
uuid = "de688a37-743e-4ac2-a6f0-bd62414d1aa7"
version = "0.6.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CFTime]]
deps = ["Dates", "Printf"]
git-tree-sha1 = "ed2e76c1c3c43fd9d0cb9248674620b29d71f2d1"
uuid = "179af706-886a-5703-950a-314cd64e0468"
version = "0.1.2"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "968c1365e2992824c3e7a794e30907483f8469a9"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.4.1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools", "SHA"]
git-tree-sha1 = "5e21a254d82c64b1a4ed9dbdc7e87c5d9cf4a686"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.10.12"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonDataModel]]
deps = ["CFTime", "DataStructures", "Dates", "Preferences", "Printf"]
git-tree-sha1 = "7f5717cbb2c1ce650cfd454451f282df33103596"
uuid = "1fbeeb36-5f17-413c-809b-666fb144f157"
version = "0.2.5"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "6e8d74545d34528c30ccd3fa0f3c00f8ed49584c"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.11"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "7cb0d72a53c1d93665eeadfa9d51af9df60bf6b2"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.10"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DiskArrays]]
deps = ["OffsetArrays"]
git-tree-sha1 = "1bfa9de80f35ac63c6c381b2d43c590875896a1f"
uuid = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
version = "0.3.22"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a6c00f894f24460379cb7136633cef54ac9f6f4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.103"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArraysCore"]
git-tree-sha1 = "499b1ca78f6180c8f8bdf1cabde2d39120229e5c"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.6"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0f478d8bad6f52573fb7658a263af61f3d96e43a"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.5.1"

[[deps.FastRounding]]
deps = ["ErrorfreeArithmetic", "LinearAlgebra"]
git-tree-sha1 = "6344aa18f654196be82e62816935225b3b9abe44"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.3.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "35f0c0f345bff2c6d636f95fdb136323b5a796ef"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.7.0"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "50351f83f95282cf903e968d7c6e8d44a5f83d0b"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "72b2e3c2ba583d1a7aa35129e56cf92e07c083e3"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.21.4"

[[deps.GenericFFT]]
deps = ["AbstractFFTs", "FFTW", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "1bc01f2ea9a0226a60723794ff86b8017739f5d9"
uuid = "a8297547-1b15-4a5a-a998-a2ac5f1cef28"
version = "0.1.6"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d53480c0793b13341c40199190f92c611aa2e93c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.2"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "f57a64794b336d4990d90f80b147474b869b1bc4"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "38c8874692d48d5440d5752d6c74b0c6b0b60739"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.2+1"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8ecb0b34472a3c98f945e3c75fc7d5428d165511"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.9.3+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "EnumX", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "f59e639916283c1d2e106d2b00910b50f4dab76c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.21.2"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "9bbb5130d3b4fa52846546bca4791ecbdfb52730"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.38"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "d65930fa2bc96b07d7691c652d701dcbe7d9cf0b"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "95063c5bc98ba0c47e75e05ae71f1fed4deac6f6"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.12"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "c879e47398a7ab671c782e02b51a4456794a7fa3"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "a84f8f1e8caaaa4e3b4c101306b9e801d3883ace"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Permutations", "Primes", "SimplePolynomials"]
git-tree-sha1 = "558a338f1eeabe933f9c2d4052aa7c2c707c3d52"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.1.12"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8a5b4d2220377d1ece13f49438d71ad20cf1ba83"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.1.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8f6af051b9e8ec597fa09d8885ed79fd582f33c9"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.10"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "6979eccb6a9edbbb62681e158443e79ecc0d056a"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.3.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Setfield", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "35fa3c150cd96fd77417a23965b7037b90d6ffc9"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.19.12"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "9b11acd07f21c4d035bd4156e789532e8ee2cc70"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.6.9"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "c8e29e2bacb98c9b6f10445227a8b0402f2f173a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test", "UnicodeFun"]
git-tree-sha1 = "8f52dbaa1351ce4cb847d95568cb29e62a307d93"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.6"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "f512dc13e64e96f703fd92ce617755ee6b5adf0f"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.8"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b01beb91d20b0d1312a9471a36017b5b339d26de"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mods]]
git-tree-sha1 = "61be59e4daffff43a8cec04b5e0dc773cbb5db3a"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "1.3.3"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.NCDatasets]]
deps = ["CFTime", "CommonDataModel", "DataStructures", "Dates", "DiskArrays", "NetCDF_jll", "NetworkOptions", "Printf"]
git-tree-sha1 = "7fcb4378f9c648a186bcb996fa29acc929a179ed"
uuid = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
version = "0.13.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetCDF]]
deps = ["DiskArrays", "Formatting", "NetCDF_jll"]
git-tree-sha1 = "328178762645783b20495d408ab317b4c2d25b1a"
uuid = "30363a11-5582-574a-97bb-aa9a979735b9"
version = "0.11.7"

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "XML2_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "10c612c81eaffdd6b7c28a45a554cdd9d2f40ff1"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "400.902.208+0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "PMIx_jll", "TOML", "Zlib_jll", "libevent_jll", "prrte_jll"]
git-tree-sha1 = "694458ae803b684f09c07f90459cb79655fb377d"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.0+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f6f85a2edb9c356b829934ad3caed2ad0ebbfc99"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.29"

[[deps.PMIx_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "Libdl", "Zlib_jll", "libevent_jll"]
git-tree-sha1 = "8b3b19351fa24791f94d7ae85faf845ca1362541"
uuid = "32165bc3-0280-59bc-8c0b-c33b6203efab"
version = "4.2.7+0"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "5ded86ccaf0647349231ed6c0822c10886d4a1ee"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.1"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4745216e94f71cb768d58330b059c9b76f32cb66"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.14+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "4f69b02cf40a0f494d0438ab29de32e14ef96e7b"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.18"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8ec28846dbf846228a32de5a6912c63e2052e3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.53"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "5a95b69396b77fdb2c48970a535610c4743810e2"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.5"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "1d05623b5952aed1307bf8b43bec8b8d1ef94b6e"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "db0219befe4507878b1a90e07820fed3e62c289d"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "b608903049d11cc557c45e03b3a53e9260579c19"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.4"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "dcc02923a53f316ab97da8ef3136e80b4543dbf1"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.0"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "d537c31cf9995236166e3e9afc424a5a1c59ff9d"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.14"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SpeedyWeather]]
deps = ["AbstractFFTs", "Adapt", "AssociatedLegendrePolynomials", "BitInformation", "CUDA", "CodecZlib", "Dates", "DocStringExtensions", "FFTW", "FLoops", "FastGaussQuadrature", "GenericFFT", "JLD2", "KernelAbstractions", "LinearAlgebra", "NetCDF", "Primes", "Printf", "ProgressMeter", "Random", "Statistics", "TOML", "UnicodePlots"]
git-tree-sha1 = "4accb398993b07dc252ab5a704c571cae796b722"
uuid = "9e226e20-d153-4fed-8a5b-493def4f21a9"
version = "0.6.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableHashTraits]]
deps = ["Compat", "SHA", "Tables", "TupleTools"]
git-tree-sha1 = "d29023a76780bb8a3f2273b29153fd00828cb73f"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "1.1.1"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "e579d3c991938fecbb225699e8f611fa3fbf2141"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.79"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "155515ed4c4236db30049ac1495e2969cc06be9d"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.4.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "PrecompileTools", "Printf", "Requires", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "b96de03092fe4b18ac7e4786bee55578d4b75ae8"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.6.0"

    [deps.UnicodePlots.extensions]
    FreeTypeExt = ["FileIO", "FreeType"]
    ImageInTerminalExt = "ImageInTerminal"
    IntervalSetsExt = "IntervalSets"
    TermExt = "Term"
    UnitfulExt = "Unitful"

    [deps.UnicodePlots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    FreeType = "b38be410-82b0-50bf-ab77-7b57e271db43"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Term = "22787eb5-b846-44ae-b979-8e399b8463ab"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eddd19a8dea6b139ea97bdc8a0e2667d4b661720"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.0.6+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevent_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenSSL_jll"]
git-tree-sha1 = "f04ec6d9a186115fb38f858f05c0c4e1b7fc9dcb"
uuid = "1080aeaf-3a6a-583e-a51c-c537b09f60ec"
version = "2.1.13+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.prrte_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "Libdl", "PMIx_jll", "libevent_jll"]
git-tree-sha1 = "5adb2d7a18a30280feb66cad6f1a1dfdca2dc7b0"
uuid = "eb928a42-fffd-568d-ab9c-3f5d54fc65b9"
version = "3.0.2+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─d8e3d937-bcda-4c84-b543-e1324f696bbc
# ╟─8ef88534-dac4-4a62-b623-dcaf63482a96
# ╟─cfb8f979-37ca-40ab-8d3c-0053911717e7
# ╟─eb95e773-b12a-40a4-a4f1-9dced86fc8a2
# ╟─75cacd05-c9f8-44ba-a0ce-8cde93fc8b85
# ╠═18ddf155-f9bc-4e5b-97dc-762fa83c9931
# ╟─87fdc7c2-536e-4aa1-9f68-8aec4bc7069d
# ╟─8d4d8b93-ebfe-41ff-8b9e-f8931a9e83c2
# ╟─25223f7b-22f7-46c2-9270-4430eb6c186e
# ╟─034fc483-b188-4b2a-891a-61b76c74072d
# ╠═039ec632-d238-4e63-81fc-a3225ccd2aee
# ╟─5d31e2a8-e357-4479-bc48-de1a1b8bc4d4
# ╟─724901e9-a19a-4d5f-aa6a-79ec0f230f24
# ╠═1431b11f-7838-41da-92e3-bcca9f4215b3
# ╠═de5d415f-8216-473d-8e0b-a73139540e1e
# ╟─15dee5d8-e995-4e5a-aceb-48bcce42e76d
# ╠═af022b36-34a9-497f-8b23-b76f5a98e741
# ╠═a93c36c9-b687-44b9-b0b6-5fe636ab061c
# ╟─2287bff1-6fb0-4431-8e15-aff3d7b6e005
# ╠═c0ff6c61-c4be-462b-a91c-0ee1395ef584
# ╟─e24e54a7-804e-40e8-818e-8766e5e3732b
# ╠═97e1ce89-f796-4bd1-8e82-94fc838829a6
# ╟─049e2164-24ac-467c-9d96-77510ac6ff57
# ╠═69531da2-5b25-453b-bc86-2db6a944e62a
# ╟─191fa774-b5f2-41c5-b913-04a4b4138af3
# ╠═5da6b1c1-4a26-4159-9386-7de456c1c697
# ╠═0ed9c5d9-8bef-4c1e-9be3-5aff1a7d226c
# ╟─b85fdf41-ef8f-4314-bc3c-383947b9f02c
# ╠═00776863-2260-48a8-83c1-3f2696f11d96
# ╟─16ca594c-c9db-4528-aa65-bab12cb6e22a
# ╟─ea517bbf-eb14-4d72-a4f4-9bb823e02f88
# ╟─140bcdac-4145-47b3-952f-bfe50f6ed41c
# ╟─849775fa-4990-47d3-afe0-d0a049bb90af
# ╠═4d517df8-0496-40a2-8e44-5beda6cd7226
# ╟─6932b969-0760-4f09-935a-478ac56de262
# ╠═1d8a69b7-52db-4865-8bf2-712c2b6442f5
# ╟─5884999f-f136-4ae7-8831-cc3a36f50a98
# ╠═f2510e0a-23f2-4a40-a7db-7b59898facfa
# ╟─4640a179-3373-4901-ac31-31022e8c7eb2
# ╟─d13c319d-345a-40b8-b90d-b0b226225434
# ╟─901548f8-a6c9-48f8-9c8f-887500081316
# ╟─590c1026-8e82-4dc7-a07b-6f3b96fbc0ee
# ╟─567fa8d3-35b4-40d7-8404-ae78d2874380
# ╟─0d8fffdc-a9f5-4d82-84ec-0f27acc04c21
# ╠═abdbbcaa-3a76-4a47-824d-6da73bc71c31
# ╟─1cef338d-5c4a-4ea5-98d7-9ac4f11922f3
# ╠═71cff056-a36c-4fd4-babb-53018894ac5c
# ╟─83be4f9d-6c85-4e5b-9379-00618c9e39be
# ╠═7c7439f0-d678-4b68-a5e5-bee650fa17e2
# ╟─d1a741ad-f28d-48c7-9f15-d55d0801573d
# ╟─a046b625-b046-4ca0-adde-be5249a420f4
# ╟─514ee86b-0aeb-42cd-b4cd-a795ed23b3de
# ╟─c98fcf26-d715-47c2-84ac-63bffe02d813
# ╟─77a73a9d-9d78-4cf5-ae19-f1107fa5b4c2
# ╠═80c72898-139e-44af-bab0-ca638f282188
# ╟─b0ca64b8-0211-4d1c-b007-7583bf8ac908
# ╟─8767e475-e842-43d6-8797-8bcbaa51b5cd
# ╠═eee25676-16d0-4479-8887-e3fe6ffe5420
# ╠═bace5438-1d2a-45aa-b3f0-1780db5857be
# ╠═4384cc1d-348d-49a8-99c7-8c304e1e5d49
# ╠═c64b1647-eece-47a8-b4c7-485e0a730507
# ╟─f720550a-6637-41ca-aaaa-8b7dd218a00f
# ╠═2d8ff967-4ed3-4cf5-a99e-786c5249ea51
# ╠═984fe3c6-b1a6-47be-8644-46ea44519286
# ╠═54622eca-6ae0-4580-a5d2-3997e2f57907
# ╠═54c03392-bf2a-4fd1-9550-287fec6c3a80
# ╠═df3a3777-5660-4785-aea6-836442329dc2
# ╠═50b6ac84-0a73-4f04-8516-ecee25b6bfcb
# ╠═f010953a-eb94-4a7a-bd03-7c6a08cfced4
# ╠═67e9bad7-6b16-450c-84d8-9b9a13cd423a
# ╟─51f3fd00-508b-4b42-bd95-ae19cb19b4db
# ╟─8afe64e3-d19a-4801-b7b8-56d886f7a59a
# ╟─ebcf224f-c006-4098-abf0-5c3644e2ee97
# ╟─73238f6c-b8d3-4f92-bdfe-1c657e239903
# ╟─247b4c3a-2777-47ed-8cc9-e90a5cdb640b
# ╟─0353c906-55d9-4419-a55d-8fcd374004d7
# ╟─1c33dc21-04af-4139-9061-696db73c3249
# ╟─70713834-3246-45a4-a4c8-68513bb853ce
# ╟─2b33a8a1-3772-4fb3-a914-a20a7aae91bc
# ╟─f7a96c67-bb62-4440-8c8a-f31ebe0aab7e
# ╟─b768707a-5077-4662-bcd1-6d38b3e4f929
# ╟─419c8c31-6408-489a-a50e-af712cf20b7e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
