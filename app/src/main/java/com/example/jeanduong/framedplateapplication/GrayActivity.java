package com.example.jeanduong.framedplateapplication;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import com.googlecode.leptonica.android.Scale;
import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.TreeSet;

import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static java.lang.Math.cos;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sin;

public class GrayActivity extends AppCompatActivity {
    static final String TAG = "OCR over gray";

    TessBaseAPI tess_engine; //Tess API reference
    String datapath = ""; //path to folder containing language data file
    String lang = "fra";

    // To be placed before onCreate!!!
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "Library loaded successfully");
                    Toast.makeText(GrayActivity.this, "OpenCV loaded successfully", Toast.LENGTH_LONG).show();
                    torture();
                } break;
                default:
                {
                    Log.i("OpenCV", "Failed to load");
                    Toast.makeText(GrayActivity.this, "Failed to load OpenCV", Toast.LENGTH_LONG).show();
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gray);
    }

    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private void torture()
    {
        ///////////
        // Setup //
        ///////////

        // Bitmap is only needed to load image from ressources and create data array.
        // It can be deleted after that, since only Mat objects will be used for processing

        Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.balser);

        final int h = img_rgb.getHeight();
        final int w = img_rgb.getWidth();
        Mat mat_rgb = new Mat(h, w, CvType.CV_8UC3);

        Utils.bitmapToMat(img_rgb, mat_rgb);
        img_rgb.recycle(); // May make the app crash
        //Imgproc.GaussianBlur(mat_rgb, mat_rgb, new Size(5, 5), 0.0); // Blur to remove some noise

        ////////////////////////////////
        // Leydier's pseudo-luminance //
        ////////////////////////////////

        Imgproc.cvtColor(mat_rgb, mat_rgb, Imgproc.COLOR_BGR2RGB); // OpenCV color image are BGR!!!
        Mat mat_lprime = MakeLPrime(mat_rgb);
        mat_rgb.release(); // May make the app crash
        //Imgproc.GaussianBlur(mat_lprime, mat_lprime, new Size(5, 5), 0.0); // Blur to remove some noise

        ////////////////////
        // MSER detection //
        ////////////////////

        MatOfKeyPoint mokp = new MatOfKeyPoint();
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.MSER);

        detector.detect(mat_lprime, mokp);

        Log.i(TAG, "|MSER| = " + mokp.rows());

        KeyPoint[] aokp = mokp.toArray();
        LinkedList<android.graphics.Rect> mser_rcts = new LinkedList<android.graphics.Rect>();
        MatOfDouble sample_sizes = new MatOfDouble(aokp.length, 1);
        double[] mser_size_array = new double[aokp.length];

        for (int k = 0; k < aokp.length; ++k)
        {
            KeyPoint kp = aokp[k];
            int half_size = (int)(kp.size / 2.0);
            Point pt = kp.pt;
            int x = (int) pt.x;
            int y = (int) pt.y;

            int l = (int) max(0, x - half_size);
            int r = (int) min(w - 1, x + half_size);
            int t = (int) max(0, y - half_size);
            int b = (int) min(h - 1, y + half_size);

            sample_sizes.put(k, 0, kp.size);

            mser_size_array[k] = kp.size;

            mser_rcts.add(new android.graphics.Rect(l, t, r, b));
            //Imgproc.rectangle(mat_lprime, new Point((int)(pt.x - half_size), (int)(pt.y - half_size)), new Point((int)(pt.x + half_size), (int)(pt.y + half_size)), new Scalar(0));
        }

        MatOfDouble sample_sizes_mean = new MatOfDouble();
        Core.meanStdDev(sample_sizes, sample_sizes_mean, new MatOfDouble());

        Arrays.sort(mser_size_array);
        double mser_size_med = mser_size_array[(int)(mser_size_array.length / 2)];

        // Gathering rectangles
/*
        LinkedList<android.graphics.Rect> zois = Gather_heuristic(mser_rcts);

        Log.i(TAG, "Remains " + zois.size() + " ZOIs");

        for (int k = 0; k < zois.size(); ++k)
        {
            android.graphics.Rect z = zois.get(k);
            Imgproc.rectangle(mat_lprime, new Point(z.left, z.top), new Point(z.right, z.bottom), new Scalar(255));
        }
*/
        ////////////////////
        // Edge detection //
        ////////////////////

        Mat mat_edges = new Mat(h, w, CvType.CV_8UC1);

        // Use Canny edge detector (keep only sites with values "around average luminance")

        MatOfDouble mu = new MatOfDouble();
        MatOfDouble sigma = new MatOfDouble();
        Core.meanStdDev(mat_lprime, mu, sigma);

        double lprime_mean = mu.get(0, 0)[0];
        double lprime_dev = sigma.get(0, 0)[0];
        double th_low = lprime_mean - lprime_dev;
        double th_high = lprime_mean + lprime_dev;

        Imgproc.Canny(mat_lprime, mat_edges, th_low, th_high);

        //////////////////////////
        // Hough line detection //
        //////////////////////////

        Mat mat_dx = new Mat(); // Gradient map in first canonical direction
        Mat mat_dy = new Mat(); // Gradient map in second canonical direction
        Mat mat_seeds = new Mat(h, w, CvType.CV_8UC1, new Scalar(0)); // Mask for Hough detector

        Imgproc.spatialGradient(mat_lprime, mat_dx, mat_dy);

        // Heuristic parameters for Hough line detector
        double distance_res = 1.0;
        double angle_res = PI / 180;
        int significant_nb_intersections = 100;
        double segment_min_length = (int)(w / 10.0);
        //double segment_max_gap = sample_sizes_mean.get(0, 0)[0]; // Average size for MSER
        double segment_max_gap = mser_size_med; // Median size for MSER

        // Seeds for gradient "rather vertical" and up oriented (negative)
        mat_seeds.setTo(new Scalar(0));

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                if (mat_edges.get(r, c)[0] > 0.0)
                    if (-mat_dy.get(r, c)[0] > 4 * abs(mat_dx.get(r, c)[0]))
                        mat_seeds.put(r, c, 255);

        Mat h_neg_segments = new Mat(h, w, CvType.CV_8UC1);

        Imgproc.HoughLinesP(mat_seeds, h_neg_segments, distance_res, angle_res, significant_nb_intersections, segment_min_length, segment_max_gap);

        // Seeds for gradient "rather vertical" and down oriented (positive)
        mat_seeds.setTo(new Scalar(0));

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                if (mat_edges.get(r, c)[0] > 0.0)
                    if (mat_dy.get(r, c)[0] > 4 * abs(mat_dx.get(r, c)[0]))
                        mat_seeds.put(r, c, 255);

        Mat h_pos_segments = new Mat(h, w, CvType.CV_8UC1);

        Imgproc.HoughLinesP(mat_seeds, h_pos_segments, distance_res, angle_res, significant_nb_intersections, segment_min_length, segment_max_gap);

        // Post-processing: force perfectly horizontal lines

        LinkedList<HorizontalSegment> negative_segments = new LinkedList<HorizontalSegment>();
        LinkedList<HorizontalSegment> positive_segments = new LinkedList<HorizontalSegment>();

        for (int k = 0 ; k < h_neg_segments.rows(); ++k)
        {
            double[] vec = h_neg_segments.get(k, 0);

             negative_segments.add(new HorizontalSegment(vec[0], vec[2], (int)((vec[1] + vec[3]) / 2.0)));
        }

        Collections.sort(negative_segments, new HorizontalSegmentOrdinateComparator());

        for (int k = 0 ; k < h_pos_segments.rows(); ++k)
        {
            double[] vec = h_pos_segments.get(k, 0);

            positive_segments.add(new HorizontalSegment(vec[0], vec[2], (int)((vec[1] + vec[3]) / 2.0)));
        }

        Collections.sort(positive_segments, new HorizontalSegmentOrdinateComparator());

        Log.i(TAG, "Hough detector : " + h_pos_segments.rows() + " positive line(s) found (" + positive_segments.size() + " horizontal)");
        Log.i(TAG, "Hough detector : " + h_neg_segments.rows() + " negative line(s) found (" + negative_segments.size() + " horizontal)");

        ////////////////////////
        // Make solid streams //
        ////////////////////////

        LinkedList<android.graphics.Rect> solid_streams = new LinkedList<android.graphics.Rect>();
        boolean[] free_flags = new boolean[h_pos_segments.rows()];

        Arrays.fill(free_flags, true);
        double ordinate_bound = h - 1;

        // Try each up-oriented segment
        for (int n = 0; n < negative_segments.size(); ++n)
        {
            HorizontalSegment neg_seg = negative_segments.get(n);
            double neg_alt = neg_seg.getOrdinate();

            if (neg_alt < ordinate_bound)
            {

                double neg_left = neg_seg.getLeft();
                double neg_right = neg_seg.getRight();


                // Search the closest down-oriented segment
                int id_closest = -1;
                double smallest_distance = h;

                for (int p = 0; p < positive_segments.size(); ++p)
                    if (free_flags[p]) {
                        HorizontalSegment pos_seg = positive_segments.get(p);
                        double pos_alt = pos_seg.getOrdinate();
                        double distance = pos_alt - neg_alt + 1;

                        if (distance > 0.0) {
                            double pos_left = pos_seg.getLeft();
                            double pos_right = pos_seg.getRight();

                            if ((distance < smallest_distance) &&
                                    (((pos_left > neg_left) && (pos_left < neg_right)) ||
                                            ((pos_right > neg_left) && (pos_right < neg_right)))) {
                                smallest_distance = distance;
                                id_closest = p;
                            }
                        }
                    }

                if ((id_closest >= 0) && (smallest_distance < 2 * mser_size_med)) {
                    HorizontalSegment pos_seg = positive_segments.get(id_closest);

                    solid_streams.add(new android.graphics.Rect(0, (int) neg_alt, w - 1,
                            (int) pos_seg.getOrdinate()));

                    free_flags[id_closest] = false;
                    ordinate_bound = pos_seg.getOrdinate();
                }
            }
        }

        Log.i(TAG, "Solid streams : " + solid_streams.size());

        LinkedList<android.graphics.Rect> tmp = Gather_intersection_ultimate(solid_streams);

        solid_streams.clear();
        solid_streams.addAll(tmp);

        for (int k = 0; k < solid_streams.size(); ++k)
        {
            android.graphics.Rect r = solid_streams.get(k);

            Imgproc.rectangle(mat_lprime, new Point(r.left, r.top),
                    new Point(r.right, r.bottom), new Scalar(255), 5);

            System.out.println(r);
        }

        Log.i(TAG, "Solid streams : " + solid_streams.size() + " (after fusion)");


        // Distance transform
/*
        Mat mat_dst = new Mat(h, w, CvType.CV_8UC1);

        Imgproc.distanceTransform(mat_bin, mat_dst, Imgproc.CV_DIST_L2, 3);
        Core.MinMaxLocResult mm = Core.minMaxLoc(mat_dst);

        int min_val = (int) mm.minVal;
        int max_val = (int) mm.maxVal;

        Log.i(TAG, "Minimal distance = " + min_val);
        Log.i(TAG, "Maximal distance = " + max_val);
*/

        ///////////////////////
        // Display new image //
        ///////////////////////

        Bitmap img_out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_lprime, img_out);
        //Utils.matToBitmap(mat_edges, img_out);
        ((ImageView) findViewById(R.id.gray_display_view_name)).setImageBitmap(img_out);

        mat_lprime.release(); // May make the app crash

        // Text extraction
        /*
        tess_engine = new TessBaseAPI();
        datapath = getFilesDir() + "/tesseract/";

        //make sure training data has been copied
        checkFile(new File(datapath + "tessdata/"));

        tess_engine.init(datapath, lang);
        tess_engine.setImage(img_out);

        String txt = tess_engine.getUTF8Text();

        tess_engine.end();

        //img_gray.recycle(); // Sometimes make the app crash
        // Print text in IDE output console
        Log.i(TAG, txt);
        */
    }

    private Mat MakeLPrime(Mat mat_rgb)
    {
        final int h = mat_rgb.rows();
        final int w = mat_rgb.cols();

        Mat mat_L_prime = new Mat(h, w, CvType.CV_8UC1);

        double red, green, blue, s, v, mi, ma;
        double[] triplet;

        // Time consuming version

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
            {
                triplet = mat_rgb.get(r, c);
                red = triplet[0];
                green = triplet[1];
                blue = triplet[2];

                ma = max(red, max(green, blue));
                mi = min(red, min(green, blue));

                if (ma > 0.0)
                    s = 255.0 * (1.0 - (mi / ma));
                else
                    s = 0.0;

                v = ((255.0 - s) * (red * 0.299 + green * 0.587 + blue * 0.114)) / 255.0;

                mat_L_prime.put(r, c, (int)min(255.0, max(0.0, v)));
            }

        return mat_L_prime;
    }

    private void copyFiles() {
        try {
            //location we want the file to be at
            String filepath = datapath + "/tessdata/" + lang + ".traineddata";

            //get access to AssetManager
            AssetManager assetManager = getAssets();

            //open byte streams for reading/writing
            InputStream instream = assetManager.open("tessdata/" + lang + ".traineddata");
            OutputStream outstream = new FileOutputStream(filepath);

            //copy the file to the location specified by filepath
            byte[] buffer = new byte[1024];
            int read;
            while ((read = instream.read(buffer)) != -1) {
                outstream.write(buffer, 0, read);
            }
            outstream.flush();
            outstream.close();
            instream.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void checkFile(File dir) {
        //directory does not exist, but we can successfully create it
        if (!dir.exists()&& dir.mkdirs()){
            copyFiles();
        }
        //The directory exists, but there is no data file in it
        if(dir.exists()) {
            String datafilepath = datapath+ "/tessdata/" + lang + ".traineddata";
            File datafile = new File(datafilepath);
            if (!datafile.exists()) {
                copyFiles();
            }
        }
    }

    private boolean Custom_proximity_heuristic(android.graphics.Rect rct_ref, android.graphics.Rect rct_other)
    {
        final double top_ref = rct_ref.top;
        final double bottom_ref = rct_ref.bottom;
        final double top_other = rct_other.top;
        final double bottom_other = rct_other.bottom;

        // Horizontal projections should overlap
        if ((bottom_other < top_ref) || (top_other > bottom_ref)) return false;

        final double height_ref = rct_ref.height();
        final double height_other = rct_other.height();
        final double vertical_range_overlap = min(bottom_ref, bottom_other) - max(top_ref, top_other) + 1;

        // Overlap range should be significant
        if (vertical_range_overlap < min(height_ref, height_other) / 2.0) return false;

        final double left_ref = rct_ref.left;
        final double right_ref = rct_ref.right;
        final double left_other = rct_other.left;
        final double right_other = rct_other.right;
        final double width_ref = rct_ref.width();
        final double width_other = rct_other.width();

        // Gap between candidates should not be greater than heights
        if (left_ref < left_other)
        {
            if (left_other - right_ref > max(height_ref, height_other))
            return false;
        }
        else if (left_ref - right_other > max(height_ref, height_other))
        return false;

        return true;
    }

    private LinkedList<android.graphics.Rect> Gather_intersection(LinkedList<android.graphics.Rect> rectangles)
    {
        LinkedList<android.graphics.Rect> rcts = new LinkedList<android.graphics.Rect>(rectangles);
        LinkedList<android.graphics.Rect> tmp = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> res = new LinkedList<android.graphics.Rect>();

        while (rcts.size() > 0)
        {
            tmp.clear();
            android.graphics.Rect z = rcts.get(0);

            for (int k = 1; k < rcts.size(); ++k)
            {
                android.graphics.Rect r = rcts.get(k);

                if (Custom_proximity_heuristic(z, r))
                    z.union(r);
                else
                    tmp.add(r);
            }

            res.add(z);
            rcts.clear();
            rcts.addAll(tmp);
        }

        return res;
    }

    private LinkedList<android.graphics.Rect> Gather_intersection_ultimate(LinkedList<android.graphics.Rect> rectangles)
    {
        LinkedList<android.graphics.Rect> rcts = new LinkedList<android.graphics.Rect>(rectangles);
        LinkedList<android.graphics.Rect> res = Gather_intersection(rectangles);

        while (res.size() != rcts.size())
        {
            rcts.clear();
            rcts.addAll(res);
            res.clear();
            res = Gather_intersection(rcts);
        }

        return res;
    }

    private LinkedList<android.graphics.Rect> Gather_alignment(LinkedList<android.graphics.Rect> rectangles)
    {
        LinkedList<android.graphics.Rect> rcts = new LinkedList<android.graphics.Rect>(rectangles);
        LinkedList<android.graphics.Rect> tmp = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> res = new LinkedList<android.graphics.Rect>();

        while (rcts.size() > 0)
        {
            tmp.clear();
            android.graphics.Rect z = rcts.get(0);

            for (int k = 1; k < rcts.size(); ++k)
            {
                android.graphics.Rect r = rcts.get(k);

                if (Custom_proximity_heuristic(z, r))
                    z.union(r);
                else
                    tmp.add(r);
            }

            res.add(z);
            rcts.clear();
            rcts.addAll(tmp);
        }

        return res;
    }

    private LinkedList<android.graphics.Rect> Gather_heuristic(LinkedList<android.graphics.Rect> rectangles)
    {
        boolean reloop = true;
        LinkedList<android.graphics.Rect> res = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> tmp = new LinkedList<android.graphics.Rect>();

        res.addAll(rectangles);

        int card_before = 0;
        int card_after = 0;

        do {
            card_before = res.size();
            tmp.clear();
            tmp.addAll(Gather_intersection(res));
            res.clear();
            res.addAll(Gather_alignment(tmp));
            card_after = res.size();

            reloop = (card_after != card_before);
        }
        while (reloop);

        return res;
    }
}
