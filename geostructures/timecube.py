import multiprocessing
import functools
import pandas as pd
import numpy as np
import random
from datetime import datetime
from math import floor, ceil
from tqdm import tqdm
import geostructures as gs
from geostructures import Coordinate
from geostructures.structures import GeoShape
from geostructures import collections as gscol
from geostructures.time import TimeInterval

import plotly.graph_objects as go


from typing import List, Tuple


def make_random_ellipses(n: int, centroid_ul: Coordinate, centroid_lr: Coordinate, smajor_bounds: Tuple[float, float], sminor_bounds: Tuple[float, float], time_bounds: TimeInterval) -> gscol.Track:
    ells = []
    for i in range(n):
        minx, miny = centroid_ul.to_float()
        maxx, maxy = centroid_lr.to_float()

        cx = random.uniform(minx, maxx)
        cy = random.uniform(miny, maxy)

        smaj = random.uniform(smajor_bounds[0], smajor_bounds[1])
        smin = random.uniform(sminor_bounds[0], sminor_bounds[1])

        rot = random.uniform(0, 180)
        t0 = time_bounds.start.timestamp()
        t1 = time_bounds.end.timestamp()
        tst = random.uniform(t0, t1)
        tend = random.uniform(t0, t1)
        if tst > tend:
            tend, tst = tst, tend
        ells.append(gs.GeoEllipse(Coordinate(cx, cy), smaj, smin, rot, dt=TimeInterval(datetime.fromtimestamp(tst), datetime.fromtimestamp(tend))))  # type: ignore

    return gscol.Track(ells)  # type: ignore


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper


def _round_down(x: float, nearest: float = 1.0) -> float:
    inv = 1 / nearest
    return floor(x * inv) / inv


def _round_up(x: float, nearest: float = 1.0) -> float:
    inv = 1 / nearest
    return ceil(x * inv) / inv


@memoize
def gs2shape(s: gs.GeoBox):
    return s.to_shapely()


def process_shape(shape_ind: Tuple[int, GeoShape], dx: float, dy: float, dt: float):
    shapeno, shape = shape_ind
    ((sh_minx, sh_maxx), (sh_miny, sh_maxy)) = shape.bounds
    sh_mint, sh_maxt = shape.start.timestamp(), shape.end.timestamp()

    minx = _round_down(sh_minx, dx)
    miny = _round_down(sh_miny, dy)
    maxx = _round_up(sh_maxx, dx)
    maxy = _round_up(sh_maxy, dy)
    mint = _round_down(sh_mint, dt)
    maxt = _round_up(sh_maxt, dt)
    shapely_shape = gs2shape(shape)

    bininfo = []
    for x in np.arange(minx, maxx, dx):
        for y in np.arange(miny, maxy, dy):
            for t in np.arange(mint, maxt, dt):
                ul = Coordinate(x, y)
                lr = Coordinate(x + dx, y + dy)
                tbin = TimeInterval(datetime.fromtimestamp(t), datetime.fromtimestamp(t + dt))
                bin = gs.GeoBox(ul, lr, dt=tbin)
                shapely_bin = gs2shape(bin)
                t_intersect = shape.dt.intersection(tbin)  # type: ignore
                area = shapely_bin.intersection(shapely_shape).area / shapely_bin.area * (0 if t_intersect is None else 1)
                
                t_int_elapsed = 0 if t_intersect is None else t_intersect.elapsed.total_seconds()
                t_frac = t_int_elapsed / tbin.elapsed.total_seconds()  # type: ignore
                volume = area * t_frac

                if area > 0:
                    entry = {
                        'x': x,
                        'y': y,
                        't': datetime.fromtimestamp(t),
                        'area': area,
                        'volume': volume,
                        'shapeno': shapeno
                    }
                    bininfo.append(entry)
    return pd.DataFrame(bininfo)


class TimeCube:
    dx: float
    dy: float
    dt: float

    # dt is in seconds (fractional ok)
    def __init__(self, shapes: gscol.Track, dx: float, dy: float, dt: float):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.cube_bounds = tuple(list(shapes.bounds) + [(shapes.start, shapes.end)])

        self.bin_bounds = (
            (_round_down(self.cube_bounds[0][0], self.dx), _round_up(self.cube_bounds[0][1], self.dx)),
            (_round_down(self.cube_bounds[1][0], self.dy), _round_up(self.cube_bounds[1][1], self.dy)),
            (datetime.fromtimestamp(_round_down(self.cube_bounds[2][0].timestamp(), self.dt)), datetime.fromtimestamp(_round_up(self.cube_bounds[2][1].timestamp(), self.dt)))
        )

        df = pd.DataFrame()
        p = multiprocessing.Pool()
        chunksize = int(len(shapes) / p._processes)
        print(f'{chunksize=}, {p._processes=}')
        partial_func = functools.partial(process_shape, dx=self.dx, dy=self.dy, dt=self.dt)
        with multiprocessing.Pool() as pool:
            df = pd.concat(tqdm(pool.imap_unordered(partial_func, enumerate(shapes), chunksize=chunksize), total=len(shapes)), ignore_index=True)
        #df = pd.concat(tqdm(p.imap(, enumerate(shapes), chunksize=chunksize), total=len(shapes)), ignore_index=True)
        self.bin_df = df.reset_index()

    def rank_coverage(self, rank_by='volume') -> pd.DataFrame:
        n_times = len(self.bin_df['t'].unique())
        ser = self.bin_df \
            .drop(['shapeno'], axis=1) \
            .groupby(['x', 'y']) \
            .agg({'area': ['sum', 'count'], 'volume': 'sum'})
        ser.columns = ser.columns.to_flat_index()
        ser = ser.rename(columns={
            ('area', 'sum'): 'area',
            ('volume', 'sum'): 'volume',
            ('area', 'count'): 'count',
        })
        df = pd.DataFrame(ser.sort_values(ascending=False, by=rank_by))
        return df.reset_index()

    def coverage_over_time(self, x: float, y: float, rank_by='volume') -> pd.DataFrame:
        x_bin = _round_down(x, self.dx)
        y_bin = _round_down(y, self.dy)
        res = self.bin_df[(np.isclose(self.bin_df['x'], x_bin)) & (np.isclose(self.bin_df['y'], y_bin))].drop(['shapeno'], axis=1)
        df = res.groupby(['t', 'x', 'y']) \
            .agg({'area': ['sum', 'count'], 'volume': 'sum'})
        df.columns = df.columns.to_flat_index()
        df = df.rename(columns={
            ('area', 'sum'): 'area',
            ('volume', 'sum'): 'volume',
            ('area', 'count'): 'count',
        })
        return df.reset_index().sort_values(['t'])

        

def _get_poly(x0, y0, x1, y1, val):
    geojd = {'type': 'FeatureCollection'}
    geojd['features'] = []
    coords = [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]
    geojd['features'].append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [coords]}})
    return {
        'sourcetype': 'geojson',
        'source': geojd,
        'below': '',
        'type': 'fill',
        'color': 'rgba(255,0,0,1)',
        'opacity': val,
        'fill_outlinecolor': 'rgba(255,0,0,0)',
    }
    
    
def _draw_layers(tc: TimeCube, rank_by: str, min_density: float):
    dn = tc.rank_coverage(rank_by)
    layers = [{
        'below': 'traces',
        'sourcetype': 'raster',
        'opacity': 0.5
    }]
    coverages = list(dn[rank_by])
    max_cov = max(coverages)
    norm_cov = [float(x) / max_cov for x in coverages]
    for i, r in tqdm(dn.iterrows(), total=len(dn)):
        x0, y0 = r['x'], r['y']
        x1, y1 = x0 + tc.dx, y0 + tc.dy
        poly = _get_poly(x0, y0, x1, y1, norm_cov[i])
        layers.append(poly)
    return layers
        
def density_plot(tc: TimeCube, rank_by: str = 'volume', min_density=0.0, width=800, height=800) -> go.Figure:
    data = [go.Scattermapbox(name='carto-positron', mode='markers')]
    layout = go.Layout(autosize=True, mapbox=dict(zoom=1, style='carto-positron'), width=width, height=height)
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_zoom=1,
        mapbox_layers=[
            {
                'below': 'traces',
                'sourcetype': 'raster',
                'opacity': 0.5
            }
        ],
    )
    ((minx, maxx), (miny, maxy), _) = tc.bin_bounds
    mylayers = _draw_layers(tc, rank_by, min_density)
    fig.update_layout(mapbox_layers=mylayers)
    fig.update_mapboxes(bounds={
        'west': minx,
        'east': maxx,
        'south': miny,
        'north': maxy
    })
    return fig
    
    