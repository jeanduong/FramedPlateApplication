package com.example.jeanduong.framedplateapplication;

import java.util.Comparator;

/**
 * Created by jeanduong on 12/12/2016.
 */

public class HorizontalSegment {
    double left;
    double right;
    double ordinate;

    public HorizontalSegment(double left, double right, double ordinate)
    {
        this.left = left;
        this.right = right;
        this.ordinate = ordinate;
    }

    public double getLeft() {
        return left;
    }

    public void setLeft(double left) {
        this.left = left;
    }

    public double getRight() {
        return right;
    }

    public void setRight(double right) {
        this.right = right;
    }

    public double getOrdinate() {
        return ordinate;
    }

    public void setOrdinate(double ordinate) {
        this.ordinate = ordinate;
    }

    @Override
    public String toString() {
        return "HorizontalSegment{" +
                "left=" + left +
                ", right=" + right +
                ", ordinate=" + ordinate +
                '}';
    }
}

