"""Crosshair + coordinate-readout utility for pyqtgraph PlotWidgets."""

import pyqtgraph as pg


def add_crosshair(pw: pg.PlotWidget, x_fmt: str = '.4g', y_fmt: str = '.4g',
                  label=None):
    """
    Attach a crosshair (two InfiniteLines) to *pw*.

    If *label* is a QLabel it is updated with 'x = ...   y = ...' on each
    mouse move.  Returns the SignalProxy — the caller must keep a reference
    to prevent garbage-collection.
    """
    vb = pw.getViewBox()

    vline = pg.InfiniteLine(angle=90, movable=False,
                            pen=pg.mkPen('#888888', width=0.8))
    hline = pg.InfiniteLine(angle=0,  movable=False,
                            pen=pg.mkPen('#888888', width=0.8))
    pw.addItem(vline, ignoreBounds=True)
    pw.addItem(hline, ignoreBounds=True)

    def _on_move(evt):
        pos = evt[0]
        if pw.sceneBoundingRect().contains(pos):
            pt = vb.mapSceneToView(pos)
            vline.setPos(pt.x())
            hline.setPos(pt.y())
            if label is not None:
                log_x, log_y = vb.state.get('logMode', [False, False])
                xv = 10 ** pt.x() if log_x else pt.x()
                yv = 10 ** pt.y() if log_y else pt.y()
                label.setText(f'x = {xv:{x_fmt}}    y = {yv:{y_fmt}}')

    proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=60, slot=_on_move)
    return proxy
