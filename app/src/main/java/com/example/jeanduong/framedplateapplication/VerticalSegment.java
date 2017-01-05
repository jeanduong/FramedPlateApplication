package com.example.jeanduong.framedplateapplication;

/**
 * Created by jeanduong on 05/01/2017.
 */

public class VerticalSegment {
    double top;
    double bottom;
    double abscissa;

    public VerticalSegment(double top, double bottom, double abscissa)
    {
        this.top = top;
        this.bottom = bottom;
        this.abscissa = abscissa;
    }

    public double getTop() {
        return top;
    }

    public void setTop(double top) {
        this.top = top;
    }

    public double getBottom() {
        return bottom;
    }

    public void setBottom(double bottom) {
        this.bottom = bottom;
    }

    public double getAbscissa() {
        return abscissa;
    }

    public void setAbscissa(double abscissa) {
        this.abscissa = abscissa;
    }

    @Override
    public String toString() {
        return "VerticalSegment{" +
                "top=" + top +
                ", bottom=" + bottom +
                ", abscissa=" + abscissa +
                '}';
    }
}
