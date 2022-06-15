# This module provides interactive plotting functions through Plotly
# that utilize SunPy Map and TimeSeries objects.
# The findpeaks function was originally created in the following SunPy example:
# https://docs.sunpy.org/en/stable/generated/gallery/time_series/timeseries_peak_finding.html

import numpy as np
from sunpy.coordinates import frames
from sunpy.coordinates.utils import get_limb_coordinates
from sunpy import config

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as colors

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsymmetricPercentileInterval


def plotly_map(self, clip_interval=(1, 99.5), color_scale=None,
               draw_grid=False, show_intensity=False, contours=None, 
               draw_limb=None, draw=False, summary=False, resample=None,
               **kwargs):
    """
    Returns a Plotly figure of a Map object. Currently, this does not work
    with submaps (use full disk maps).

    Parameters
    ----------
    clip interval : `tuple`
        The data is clipped to the percentile interval bounded by the provided
        values.
    color_scale : `list`
        A list of rgb strings or CSS colors. For example:
        ['rgb(0, 0, 0)', 'rgb(209, 164, 32)']
        Here is Plotly's built-in color scales:
        https://plotly.com/python/builtin-colorscales/
    draw_grid : `bool`
        Overlays heliographic Stonyhurst grid. Hover info also includes
        heliographic Carrington coordinates.
    show_intensity : `bool`
        Includes the pixel intensities of the original data in the plotly hover
        template. Avoid using this with maps greater than (1024,1024) in size.
    contours : `float`
        Extracts contours using the Map contours() method.
    draw_limb : `bool` or `sunpy.coordinates`
        Setting equal to True overlays the solar limb in the map frame.
        Alternatively, set this equal to the observer of another Map to overlay
        the solar limb from that observer's perspective.
    draw : `bool`
        Draw allows you to draw simple shapes on the top of the figure.
    resample : `list`
        Resample will reduce the angular resolution of a map. Set this equal to
        a list of the desired resolution (e.g., [1024, 1024]). 

    Returns
    -------
    plotly figure object
    """
    # Resample the map to a lower resolution for a more responsive interactive
    # plot. For reference, sample data maps are [1024, 1024] pixels.
    if resample is not None:
        self = self.resample(resample * u.pixel)
    
    bs_val=True
    # binary_string = True converts the image to a png base64 string that 
    # is then traced as a black and white image. 
    # Warning: assigning a colorscale to a large image will increase the
    # render time.
    if color_scale is not None:
        bs_val=False
    clip_interval = clip_interval*u.percent
    clip_percentages = clip_interval.to('%').value
    vmin, vmax = AsymmetricPercentileInterval(*clip_percentages).get_limits(self.data)

    bx = self.bottom_left_coord.Tx.value
    by = self.bottom_left_coord.Ty.value
    tx = self.top_right_coord.Tx.value
    ty = self.top_right_coord.Ty.value

    scalex = (tx-bx)/self.data.shape[0]
    scaley = (ty-by)/self.data.shape[1]

    Xax = np.arange(bx,tx,scalex)
    Yax = np.arange(by,ty,scaley)

    # Define native frame and functions for drawing HGS grid
    native_frame = self.center.frame
    rsun = self.center.rsun
    reference_dist = np.sqrt(self.center.frame.observer.radius**2 - rsun**2)

    # Resolution of the HGS grid lines
    resolution = 200

    def meridian(deg):
        lon = SkyCoord(np.ones(resolution) * deg*u.deg, 
                       np.linspace(-90, 90, resolution) * u.deg, 
                       frame=frames.HeliographicStonyhurst)
        lonHPC = lon.transform_to(native_frame)
        is_visible = lonHPC.spherical.distance <= reference_dist
        vis = np.where(is_visible == True)
        lonHPC = lonHPC[vis]
        lonHGC = lonHPC.heliographic_carrington.to_string()
        lonHGC = [s.replace(' ',', ') for s in lonHGC]
        lon = lon[vis]
        lon = lon.to_string()
        lon = [s.replace(' ',', ') for s in lon]
        lon_data = np.stack((lon, lonHGC),axis=-1)
        return lonHPC, lon_data
    
    def parallel(deg):
        lat = SkyCoord(np.linspace(-90, 90, resolution) * u.deg, 
                       np.ones(resolution) * deg*u.deg, 
                       frame=frames.HeliographicStonyhurst)
        latHPC = lat.transform_to(native_frame)
        is_visible = latHPC.spherical.distance <= reference_dist
        vis = np.where(is_visible == True)
        latHPC = latHPC[vis]
        latHGC = latHPC.heliographic_carrington.to_string()
        latHGC = [s.replace(' ',', ') for s in latHGC]
        lat = lat[vis]
        lat = lat.to_string()
        lat = [s.replace(' ',', ') for s in lat]
        lat_data = np.stack((lat, latHGC),axis=-1)
        return latHPC, lat_data
   
    name = r'{} {} {}'.format(self.detector, self.wavelength,
                              self.date.value.replace('T',' '))
    if summary is True:
        # Note that subplots specs requires the plot type to be image
        # not imshow. Imshow uses the underlying Image and Heatmap
        # classes of Graph Objects.
        fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                            specs=[[{"type": "table"},
                                    {"type": "image"}]])
        dt = self.exposure_time.to_string()
        wave = self.wavelength.to_string()
        measurement = self.measurement.to_string()

        dt = 'Unknown' if dt is None else dt
        wave = 'Unknown' if wave is None else wave
        measurement = 'Unknown' if measurement is None else measurement
        TIME_FORMAT = config.get("general", "time_format")
        
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=[
                        [
                            "<b>Observatory</b>",
                            "<b>Instrument</b>",
                            "<b>Detector</b>",
                            "<b>Measurement</b>",
                            "<b>Wavelength</b>",
                            "<b>Observation Date</b>",
                            "<b>Exposure Time</b>",
                            "<b>Dimension</b>",
                            "<b>Coordinate System</b>",
                            "<b>Scale</b>",
                            "<b>Reference Pixel</b>",
                            "<b>Reference Coord</b>",
                        ],
                        [self.observatory, self.instrument,
                         self.detector, measurement,
                         wave, self.date.strftime(TIME_FORMAT), 
                         dt, u.Quantity(self.dimensions).to_string(),
                         u.Quantity(self.scale).to_string(),
                         self._coordinate_frame_name,
                         u.Quantity(self.reference_pixel).to_string(),
                         u.Quantity((self._reference_longitude,
                                     self._reference_latitude)).to_string()],
                    ],
                    align="right",
                )
            ),
            row=1,
            col=1,
        )
        fig.add_trace(px.imshow(self.data, x=Xax, y=Yax,
                                zmin=vmin, zmax=vmax,
                                binary_string=bs_val,
                                labels={'x':'Helioprojective Longitude (arcsec)',
                                        'y':'Helioprojective Latitude (arcsec)'},
                                title=name).data[0],
                      row=1, col=2)
        fig.update_layout(coloraxis_showscale=False, height=700)
        fig.update_traces(col=2, hovertemplate="HPC Longitude (arcsec): %{x:.2f} <br> HPC Latitude (arcsec): %{y:.2f} <extra></extra>")
        fig.update_xaxes(col=2, title='Helioprojective Longitude (arcsec)')
        fig.update_yaxes(col=2, title='Helioprojective Latitude (arcsec)',
                         autorange=True)
    else:
        fig = px.imshow(self.data, x=Xax, y=Yax,
                        zmin=vmin, zmax=vmax,
                        binary_string=bs_val,
                        color_continuous_scale=color_scale,
                        labels={'x':'Helioprojective Longitude (arcsec)',
                                'y':'Helioprojective Latitude (arcsec)'},
                        title=name, origin='lower', **kwargs)
        fig.update_layout(coloraxis_showscale=False, height=600, width=600)
        fig.update_traces(hovertemplate="HPC Longitude (arcsec): %{x:.2f} <br> HPC Latitude (arcsec): %{y:.2f} <extra></extra>")
    
    if draw_grid is True:
        lines = np.array([-90, -70, -50, -37.5, -25, -12.5, 0,
                          12.5, 25, 37.5, 50, 70, 90])
        for i in lines:
            if i == 0:
                width = 2
                opacity = 0.7
            else:
                width = 1
                opacity = 0.5
            lonHPC, lon_cdata = meridian(i)
            fig.add_trace(go.Scatter(x=lonHPC.Tx.to_value(), y=lonHPC.Ty.to_value(), 
                                     line=dict(color='white', width=width), 
                                     opacity=opacity, customdata=lon_cdata, 
                                     hovertemplate="HPC (arcsec): (%{x:.2f}, %{y:.2f}) <br> HGS (deg): %{customdata[0]} <br> HGC (deg): %{customdata[1]} <extra></extra>"))
            latHPC, lat_cdata = parallel(i)
            fig.add_trace(go.Scatter(x=latHPC.Tx.to_value(), y=latHPC.Ty.to_value(), 
                                     line=dict(color='white', width=width), 
                                     opacity=opacity, customdata=lat_cdata, 
                                     hovertemplate="HPC (arcsec): (%{x:.2f}, %{y:.2f}) <br> HGS (deg): %{customdata[0]} <br> HGC (deg): %{customdata[1]} <extra></extra>"))
    fig.update_layout(showlegend=False)
    if show_intensity is True:
        intense = self.data
        fig.update(data=[{'customdata': intense,
                          'hovertemplate': "HPC Lon (arcsec): %{x:.2f} <br>HPC Lat (arcsec): %{y:.2f} <br>intensity: %{customdata:.2f} <extra></extra>"}])
    if contours is not None:
        contour = self.contour(contours * u.ct)
        for coords in contour:
            fig.add_trace(go.Scatter(x=coords.Tx.value, y=coords.Ty.value,
                                     mode='lines',
                                     line=dict(color='red', width=1.5),
                                     showlegend=False))
    if draw_limb is not None:
        if draw_limb is True:
            observer = self.center.observer
            limb = get_limb_coordinates(observer,
                                        self.center.rsun, 200)
            limb = limb.transform_to(native_frame)
            fig.add_trace(go.Scatter(x=limb.Tx.value, y=limb.Ty.value,
                                     line=dict(color='blue', width=1.5),
                                     hovertemplate="HPC Lon (arcsec): %{x:.2f} <br>HPC Lat (arcsec): %{y:.2f}<extra></extra>"
                                     )
                          )
        else:
            observer = draw_limb
            limb = get_limb_coordinates(observer,
                                        self.center.rsun, 200)
            limb=limb.transform_to(self.center.frame)
            rsun = self.center.rsun
            reference_dist = np.sqrt(self.center.frame.observer.radius**2 - rsun**2)
            is_visible = limb.spherical.distance <= reference_dist
            vis = np.where(is_visible == True)
            visible = limb[vis]
            hid = np.where(is_visible == False)
            hidden = limb[hid]
            fig.add_trace(go.Scatter(x=visible.Tx.value, y=visible.Ty.value,
                                     mode='markers',
                                     marker=dict(color='blue', size=4),
                                     hovertemplate="HPC Lon (arcsec): %{x:.2f} <br>HPC Lat (arcsec): %{y:.2f}<extra></extra>"
                                     )
                          )
            fig.add_trace(go.Scatter(x=hidden.Tx.value, y=hidden.Ty.value,
                                     mode='markers', opacity=0.6,
                                     marker=dict(color='blue', size=4),
                                     hovertemplate="HPC Lon (arcsec): %{x:.2f} <br>HPC Lat (arcsec): %{y:.2f}<extra></extra>"
                                     )
                          )
    if draw is True:
        fig.update_layout(
        dragmode='drawrect',
        newshape=dict(line_color='white'))
        return fig.show(config={'modeBarButtonsToAdd':['drawline',
                                                'drawopenpath',
                                                'drawclosedpath',
                                                'drawcircle',
                                                'drawrect',
                                                'eraseshape'
                                               ]})
    return fig.show()

def plotly_ts(self, gradient=False, peaks=None, **kwargs):
    """
    A Plotly figure of each channel in the TimeSeries object.

    Parameters
    ----------
    gradient : `bool`
        Uses numpy.gradient() on each channel and plots the results.
    peaks : `float`
        Activates the findpeaks function. You must set it
        equal to the DELTA value.

    Returns
    -------
    plotly figure object
    """
    dat = self.to_dataframe()
    channels = self.columns
    fig = go.Figure()
    if peaks is not None:
        for i in channels:
            fig.add_trace(
                go.Scatter(
                    x=dat.index,
                    y=dat[i],
                    name=i,
                    **kwargs
                )
            )
            minp, maxp = findpeaks(dat[i],DELTA=peaks)
            xmin, ymin = zip(*minp)
            xmax, ymax = zip(*maxp)
            fig.add_trace(go.Scatter(
                x=xmax,
                y=ymax,
                mode='markers',
                marker=dict(
                    size=8,
                    color='green',
                    symbol='circle'
                    ),
                name='Max '+i
                )
            )
            fig.add_trace(go.Scatter(
                x=xmin,
                y=ymin,
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='circle'
                    ),
                name='Min '+i
                )
            )
    elif gradient is True:
        for i in channels:
            y = np.gradient(dat[i].values)
            fig.add_trace(
                go.Scatter(
                    x=dat.index,
                    y=y,
                    name=i,
                    **kwargs
                )
            )
        unit = self.units[channels[0]]/u.second
        fig.update_yaxes(title=unit.to_string())
        fig.update_layout(yaxis_exponentformat="power",
                          hovermode="x")
        return fig.show()
    else:
        for i in channels:
            fig.add_trace(
                go.Scatter(
                    x=dat.index,
                    y=dat[i],
                    name=i,
                    **kwargs
                )
            )
    fig.update_yaxes(type="log",title=self.units[channels[0]].to_string())
    fig.update_layout(yaxis_dtick="1",yaxis_exponentformat="power",
                      hovermode="x")
    return fig.show()

def ts_summary(self):
    """
    Produces an interactive data summary for a TimeSeries object.
    The plots here are identical to those created in the
    _repr_html_ method.
    
    Returns
    -------
    plotly figure object
    """
    # Extract and build the table information
    obs = self.observatory
    if obs is None:
        try:
            obs = self.meta.metadata[0][2]["telescop"]
        except KeyError:
            obs = "Unknown"
    try:
        inst = self.meta.metadata[0][2]["instrume"]
    except KeyError:
        inst = "Unknown"
    try:
        link = f"""<a href="{self.url}" target="_blank">{inst} information</a>"""
    except AttributeError:
        link = None

    samp = self.shape[0]
    dat = self.to_dataframe()
    start = dat.index.min().round("s")
    end = dat.index.max().round("s")
    drange = dat.max() - dat.min()
    drange = drange.to_string(float_format="{:.2E}".format)
    drange = drange.replace("\n", "<br>")

    center = self.time_range.center.value.astype('datetime64[s]')
    center = str(center).replace("T", " ")
    resolution = round(self.time_range.seconds.value/self.shape[0], 3)
    resolution = str(resolution)+" s"

    channels = self.columns
    channels2 = "<br>".join(channels)

    uni = list(set(self.units.values()))
    uni = [x.unit if type(x) == u.quantity.Quantity else x for x in uni]
    uni = ["dimensionless" if x == u.dimensionless_unscaled else x for x in uni]
    uni = "<br>".join(str(x) for x in uni)

    # Define color list so each channel has matching colors in its
    # timeseries and histogram. The perm() function is necessary for
    # designating the visibility of plots using the dropdown menu.
    cols = colors.DEFAULT_PLOTLY_COLORS + colors.qualitative.Safe

    def perm(ind):
        P = [True, True] + [False for i in range(2 * len(channels) - 2)]
        if ind == 0:
            return P
        else:
            Pnew = []
            for i in range(len(P)):
                Pnew.append(P[i - 2 * ind])
            return Pnew

    # Initialize the plot, then create the timeseries and histograms.
    # Bin size is set to Scott's rule.
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "table", "rowspan": 2}, {"type": "scatter"}],
            [None, {"type": "histogram"}],
        ],
    )
    for i in range(len(channels)):
        if i == 0:
            vis = True
        else:
            vis = False
        fig.add_trace(
            go.Scatter(
                x=dat.index,
                y=dat[channels[i]],
                name=channels[i],
                marker=dict(color=cols[i]),
                visible=vis,
            ),
            row=1,
            col=2,
        )
        # Custom bin sizing slows down plotly's renderer a lot.
        # So, datasets with over 10 channels are set to use plotly's
        # default bin algorithm, which renders faster.
        if len(self.columns) < 10:
            binsize = astropy.stats.scott_bin_width(dat[channels[i]].values)
        else:
            binsize = 0
        fig.add_trace(
            go.Histogram(
                x=dat[channels[i]].values,
                name=channels[i],
                marker_color=cols[i],
                xbins=dict(size=binsize),
                showlegend=False,
                visible=vis,
            ),
            row=2,
            col=2,
        )
    Menu = []
    for i in range(len(channels)):
        Menu.append(
            dict(
                label=channels[i],
                method="update",
                args=[
                    {"visible": perm(i)},
                    {
                        "yaxis.title": str(self.units[self.columns[i]]),
                        "xaxis2.title": str(self.units[self.columns[i]]),
                    },
                ],
            ),
        )
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                #showactive=False,
                buttons=list(Menu),
                x=0,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ]
    )
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[
                    [
                        "<b>Observatory</b>",
                        "<b>Instrument</b>",
                        "<b>Channel(s)</b>",
                        "<b>Start Date</b>",
                        "<b>End Date</b>",
                        "<b>Center Date</b>",
                        "<b>Resolution</b>",
                        "<b>Samples per Channel</b>",
                        "<b>Data Range(s)</b>",
                        "<b>Units</b>",
                    ],
                    [obs, inst, channels2, start, end, center, 
                     resolution, samp, drange, uni],
                ],
                align="right",
            )
        ),
        row=1,
        col=1,
    )
    fig["layout"]["yaxis2"]["title"] = "# of occurences"
    fig["layout"]["yaxis"]["tickformat"] = ".1e"
    fig["layout"]["xaxis2"]["tickformat"] = ".1e"

    fig.update_layout(height=700, hovermode="x", showlegend=False)
    fig.update_yaxes(type="log")
    if link is not None:
        fig.add_annotation(
            xref="paper", x="0", yref="paper", y="-0.1", text=link,
            showarrow=False
        )
    return fig.show()

def findpeaks(series, DELTA):
    """
    Finds extrema in a pandas series data.

    Parameters
    ----------
    series : `pandas.Series`
        The data series from which we need to find extrema.

    DELTA : `float`
        The minimum difference between data values that defines a peak.

    Returns
    -------
    minpeaks, maxpeaks : `list`
        Lists consisting of pos, val pairs for both local minima points and
        local maxima points.
    """
    # Set inital values
    mn, mx = np.Inf, -np.Inf
    minpeaks = []
    maxpeaks = []
    lookformax = True
    start = True
    # Iterate over items in series
    for time_pos, value in series.iteritems():
        if value > mx:
            mx = value
            mxpos = time_pos
        if value < mn:
            mn = value
            mnpos = time_pos
        if lookformax:
            if value < mx-DELTA:
                # a local maxima
                maxpeaks.append((mxpos, mx))
                mn = value
                mnpos = time_pos
                lookformax = False
            elif start:
                # a local minima at beginning
                minpeaks.append((mnpos, mn))
                mx = value
                mxpos = time_pos
                start = False
        else:
            if value > mn+DELTA:
                # a local minima
                minpeaks.append((mnpos, mn))
                mx = value
                mxpos = time_pos
                lookformax = True
    # check for extrema at end
    if value > mn+DELTA:
        maxpeaks.append((mxpos, mx))
    elif value < mx-DELTA:
        minpeaks.append((mnpos, mn))
    return minpeaks, maxpeaks
