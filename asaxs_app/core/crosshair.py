"""Crosshair + coordinate-readout utility for pyqtgraph PlotWidgets."""

import pyqtgraph as pg


def add_crosshair(pw: pg.PlotWidget, x_fmt: str = '.4g', y_fmt: str = '.4g'):
    """
    Attach a crosshair (two InfiniteLines) and a coordinate TextItem to *pw*.

    The TextItem sits at the bottom-left of the live view area and updates
    as the user zooms.  Returns the SignalProxy — the caller must keep a
    reference to prevent garbage-collection.

    Usage
    -----
    self._ch_proxy = add_crosshair(self._pw_iq)
    """
    vb = pw.getViewBox()

    vline = pg.InfiniteLine(angle=90, movable=False,
                            pen=pg.mkPen('#888888', width=0.8))
    hline = pg.InfiniteLine(angle=0,  movable=False,
                            pen=pg.mkPen('#888888', width=0.8))
    pw.addItem(vline, ignoreBounds=True)
    pw.addItem(hline, ignoreBounds=True)

    coord = pg.TextItem(text='', color=(180, 180, 180), anchor=(0, 0))
    coord.setZValue(200)
    pw.addItem(coord, ignoreBounds=True)

    def _reanchor():
        xr, yr = vb.viewRange()
        # place text just inside bottom-left
        coord.setPos(xr[0] + 0.01 * (xr[1] - xr[0]),
                     yr[0] + 0.02 * (yr[1] - yr[0]))

    vb.sigRangeChanged.connect(lambda *_: _reanchor())

    def _on_move(evt):
        pos = evt[0]
        if pw.sceneBoundingRect().contains(pos):
            pt = vb.mapSceneToView(pos)
            vline.setPos(pt.x())
            hline.setPos(pt.y())
            _reanchor()
            coord.setText(f'x = {pt.x():{x_fmt}}    y = {pt.y():{y_fmt}}')

    proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=60, slot=_on_move)
    return proxy
