package com.example.jeanduong.framedplateapplication;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

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
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.opencv.core.Core.bitwise_not;
import static org.opencv.imgproc.Imgproc.MORPH_RECT;
import static org.opencv.imgproc.Imgproc.connectedComponents;

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

    public enum Field_type
    {
        SYMBOLE, SERIAL, MANUFACTURING, WARRANTY, COMMISSION
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

        MatOfKeyPoint mat_mser_keypoints = new MatOfKeyPoint();
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.MSER);

        detector.detect(mat_lprime, mat_mser_keypoints);

        Log.i(TAG, "MSER ZOIs : " + mat_mser_keypoints.rows());

        KeyPoint[] array_mser_keypoints = mat_mser_keypoints.toArray();
        LinkedList<Rect> mser_zois = new LinkedList<Rect>();
        MatOfDouble mser_radii_sample = new MatOfDouble(array_mser_keypoints.length, 1);
        double[] mser_radii_array = new double[array_mser_keypoints.length];

        for (int k = 0; k < array_mser_keypoints.length; ++k)
        {
            KeyPoint kp = array_mser_keypoints[k];
            int half_size = (int)(kp.size / 2.0);
            Point pt = kp.pt;
            int x = (int) pt.x;
            int y = (int) pt.y;

            int l = (int) max(0, x - half_size);
            int r = (int) min(w - 1, x + half_size);
            int t = (int) max(0, y - half_size);
            int b = (int) min(h - 1, y + half_size);

            mser_radii_sample.put(k, 0, kp.size);
            mser_radii_array[k] = kp.size;
            mser_zois.add(new Rect(l, t, r, b));
        }

        MatOfDouble sample_sizes_mean = new MatOfDouble();
        Core.meanStdDev(mser_radii_sample, sample_sizes_mean, new MatOfDouble());

        Arrays.sort(mser_radii_array);
        double mser_size_med = mser_radii_array[(int)(mser_radii_array.length / 2)];

        // Gathering rectangles

        mser_zois = Gather_heuristic(mser_zois);

        Log.i(TAG, "MSER ZOIs : " + mser_zois.size() + " (after fusion)");

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

        Log.d(TAG, "Hough detector : " + h_pos_segments.rows() + " positive line(s) found (" + positive_segments.size() + " horizontal)");
        Log.d(TAG, "Hough detector : " + h_neg_segments.rows() + " negative line(s) found (" + negative_segments.size() + " horizontal)");

        ////////////////////////
        // Make solid streams //
        ////////////////////////

        LinkedList<Rect> solid_streams = new LinkedList<Rect>();
        boolean[] free_flags = new boolean[h_pos_segments.rows()];

        Arrays.fill(free_flags, true);
        double ordinate_floor = 0;

        // Try each up-oriented segment
        for (int n = 0; n < negative_segments.size(); ++n)
        {
            HorizontalSegment neg_seg = negative_segments.get(n);
            double neg_ord = neg_seg.getOrdinate();

            if (neg_ord > ordinate_floor)
            {
                double neg_left = neg_seg.getLeft();
                double neg_right = neg_seg.getRight();

                // Search the closest down-oriented segment
                int id_closest = -1;
                double smallest_distance = Double.POSITIVE_INFINITY;

                for (int p = 0; p < positive_segments.size(); ++p)
                {
                    if (free_flags[p])
                    {
                        HorizontalSegment pos_seg = positive_segments.get(p);
                        double pos_ord = pos_seg.getOrdinate();

                        if (pos_ord > neg_ord)
                        {
                            double distance = pos_ord - neg_ord + 1;
                            double pos_left = pos_seg.getLeft();
                            double pos_right = pos_seg.getRight();

                            if ((distance < smallest_distance) &&
                                    (((pos_left > neg_left) && (pos_left < neg_right)) ||
                                            ((pos_right > neg_left) && (pos_right < neg_right))))
                            {
                                smallest_distance = distance;
                                id_closest = p;
                            }
                        } else
                            free_flags[p] = false;
                    }
                }

                // Found candidate
                if ((id_closest >= 0) && (smallest_distance < 2 * mser_size_med))
                {
                    HorizontalSegment pos_seg = positive_segments.get(id_closest);
                    int pos_ord = (int)pos_seg.getOrdinate();

                    solid_streams.add(new Rect(0, (int) neg_ord, w - 1, pos_ord));
                    free_flags[id_closest] = false;
                    ordinate_floor = pos_ord;
                }
            }
        }

        Log.i(TAG, "Solid streams : " + solid_streams.size());

        solid_streams = Gather_intersection_ultimate(solid_streams);

        Log.i(TAG, "Solid streams : " + solid_streams.size() + " (after fusion)");

        ///////////////////////////////////
        // Combination with MSER regions //
        ///////////////////////////////////

        int[] overlap_areas = new int[mser_zois.size()];
        int[] overlap_argmax = new int[mser_zois.size()];

        Arrays.fill(overlap_areas, 0);
        Arrays.fill(overlap_argmax, 0);

        for (int s = 1; s < solid_streams.size(); ++s)
        {
            Rect r = new Rect(0, solid_streams.get(s - 1).bottom, w - 1, solid_streams.get(s).top);

            for (int m = 0; m < mser_zois.size(); ++m)
            {
                Rect z = new Rect(mser_zois.get(m));

                if (z.intersect(r))
                {
                    int area = z.width() * z.height();

                    if (area > overlap_areas[m])
                    {
                        overlap_areas[m] = area;
                        overlap_argmax[m] = s - 1;
                    }
                }
            }
        }

        LinkedList<Rect> text_zois = new LinkedList<Rect>();

        for (int m = 0; m < mser_zois.size(); ++m)
        {
            Rect rct = mser_zois.get(m);
            int id = overlap_argmax[m];

            text_zois.add(new Rect(rct.left, solid_streams.get(id).bottom,
                    rct.right, solid_streams.get(id + 1).top));
        }

        /////////////////////////////////////////////////
        // Binarization thresholds along solid streams //
        /////////////////////////////////////////////////

        Collections.sort(solid_streams, new RectTopComparator());

        LinkedList<Rect> extended_streams = new LinkedList<Rect>();
        LinkedList<double[]> local_thresholds = new LinkedList<double[]>();

        for (int k = 0; k < solid_streams.size(); ++k)
        {
            Rect str = solid_streams.get(k);
            double[] thresholds = new double[w];

            Arrays.fill(thresholds, 0);

            // Extend streams to double thickness if possible

            final int height = str.height();
            final int half_height = (int) (height / 2.0);
            final int double_height = height * 2;
            int original_upper_bound = str.top;
            int original_lower_bound = str.bottom;
            int extended_upper_bound = original_upper_bound;
            int extended_lower_bound = original_lower_bound;

            if (original_upper_bound > half_height) extended_upper_bound -= half_height;
            else extended_upper_bound = 0;

            extended_lower_bound += double_height - (extended_lower_bound - extended_upper_bound + 1);
            extended_lower_bound = min(h - 1, extended_lower_bound);
            extended_streams.add(new Rect(0, extended_upper_bound, w - 1, extended_lower_bound));

            // Minimal and maximal luminance values within extended stream to "seed" bipartition

            double seed_dark = Double.POSITIVE_INFINITY;
            double seed_bright = 0.0;

            for (int r = extended_upper_bound; r <= extended_lower_bound; ++r)
                for (int c = 0; c < w; ++c)
                {
                    final double val = mat_lprime.get(r, c)[0];

                    seed_dark = min(seed_dark, val);
                    seed_bright = max(seed_bright, val);
                }

            // Threshold computation for each abscissa using 2-means
            // Abscissa where computation failed re-processed further

            for (int c = 0; c < w; ++c)
            {
                int nb_loops = 1;
                int nb_dark = 0;
                int nb_bright = 0;
                double cumul_dark = 0.0;
                double cumul_bright = 0.0;

                double th = 0.0;

                // First loop of 2_means
                for (int r = extended_upper_bound; r <= extended_lower_bound; ++r)
                {

                    final double val = mat_lprime.get(r, c)[0];

                    if (abs(val - seed_dark) < abs(val - seed_bright))
                    {
                        cumul_dark += val;
                        ++nb_dark;
                    }
                    else
                    {
                        cumul_bright += val;
                        ++nb_bright;
                    }
                }

                // Not everything ending in one class -> more loops needed
                if ((nb_dark > 0) && (nb_bright > 0))
                {
                    double centroid_dark  = cumul_dark / nb_dark;
                    double centroid_bright = cumul_bright / nb_bright;
                    double new_th = (centroid_dark + centroid_bright) / 2.0;

                    while ((new_th != th) && (nb_loops < double_height))
                    {
                        th = new_th;

                        nb_dark = 0;
                        nb_bright = 0;
                        cumul_dark = 0.0;
                        cumul_bright = 0.0;

                        for (int r = extended_upper_bound; r <= extended_lower_bound; ++r)
                        {
                            final double val = mat_lprime.get(r, c)[0];

                            if (abs(val - centroid_dark) < abs(val - centroid_bright))
                            {
                                cumul_dark += val;
                                ++nb_dark;
                            }
                            else
                            {
                                cumul_bright += val;
                                ++nb_bright;
                            }
                        }

                        centroid_dark = cumul_dark / nb_dark;
                        centroid_bright = cumul_bright / nb_bright;
                        new_th = (centroid_dark + centroid_bright) / 2.0;

                        ++nb_loops;
                    }

                seed_dark = centroid_dark;
                seed_bright = centroid_bright;
                }

                thresholds[c] = th;
            }

            // Right (resp. left) bound for leftmost (resp. rightmost)
            // residues, i.e. portions where threshold computation failed
            int c_left = 0;
            int c_right = w - 1;

            while (thresholds[c_left] == 0.0) ++c_left;
            while (thresholds[c_right] == 0.0) --c_right;

            for (int c = 0; c < c_left; ++c) thresholds[c] = thresholds[c_left];
            for (int c = c_right + 1; c < w; ++c) thresholds[c] = thresholds[c_right];

            // Estimate thresholds for residues using linear interpolation

            boolean within_residue = false;
            int portion_left_bound = c_left; // 'int' safer for substraction

            for (int c = c_left; c <= c_right; ++c)
            {
                if ((thresholds[c] == 0.0) && !within_residue)
                {
                    within_residue = true;
                    portion_left_bound = c;
                }
                else if ((thresholds[c] != 0.0) && within_residue)
                {
                    within_residue = false;

                    final double delta = (thresholds[c - 1] - thresholds[portion_left_bound]) / (c - 1 - portion_left_bound + 1);

                    for (int cc = portion_left_bound; cc < c; ++cc)
                        thresholds[cc] = thresholds[cc - 1] + delta;
                }
            }

            local_thresholds.add(thresholds);
        }

        ///////////////////////////////////////////
        // Binarization between extended streams //
        ///////////////////////////////////////////

        Mat mat_bin = new Mat(h, w, CvType.CV_8UC1);

        for (int k = 1; k < extended_streams.size(); ++k)
        {
            // Bounds for region to binarize
            final int upper_bound = extended_streams.get(k - 1).bottom;
            final int lower_bound = extended_streams.get(k).top;
            // Local binarization thresholds
            final double[] th_upper = local_thresholds.get(k - 1);
            final double[] th_lower = local_thresholds.get(k);

            for (int c = 0; c < w; ++c)
            {
                final double th = (th_upper[c] + th_lower[c]) / 2.0;

                for (int r = upper_bound; r <= lower_bound; ++r)
                    if (mat_lprime.get(r, c)[0] < th) mat_bin.put(r, c, 0);
                    else mat_bin.put(r, c, 255);
            }
        }

        Mat mask = new Mat(h, w, CvType.CV_8UC1, new Scalar(0));

        for (int k = 0; k < text_zois.size(); ++k)
        {
            Rect rct = text_zois.get(k);

            for (int r = rct.top; r <= rct.bottom; ++r)
                for (int c = rct.left; c <= rct.right; ++c)
                    mask.put(r, c, 255);
        }

        Mat mat_frankenstein = new Mat(h, w, CvType.CV_8UC1, new Scalar(255));

        mat_bin.copyTo(mat_frankenstein, mask);

        mat_frankenstein.copyTo(mat_bin);
        bitwise_not(mat_frankenstein, mat_frankenstein);

        Mat lateralStructure = Imgproc.getStructuringElement(MORPH_RECT, new Size(3 * (int)mser_size_med, 1));

        Imgproc.erode(mat_frankenstein, mat_frankenstein, lateralStructure, new Point(-1, -1), 1);
        Imgproc.dilate(mat_frankenstein, mat_frankenstein, lateralStructure, new Point(-1, -1), 1);

        bitwise_not(mat_frankenstein, mask);

        mat_bin.copyTo(mat_frankenstein, mask);

        ///////////////////////
        // Display new image //
        ///////////////////////

        Bitmap img_out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(mat_lprime, img_out);
        //Utils.matToBitmap(mat_edges, img_out);
        //Utils.matToBitmap(mat_bin, img_out);
        //Utils.matToBitmap(mask, img_out);
        Utils.matToBitmap(mat_frankenstein, img_out);

        ((ImageView) findViewById(R.id.gray_display_view_name)).setImageBitmap(img_out);

        //////////////////////////////
        // Run OCR and extract data //
        //////////////////////////////

        tess_engine = new TessBaseAPI();
        datapath = getFilesDir() + "/tesseract/";

        //make sure training data has been copied
        checkFile(new File(datapath + "tessdata/"));

        tess_engine.init(datapath, lang);
        tess_engine.setImage(img_out);

        // Arrange zones to retrieve text lines (with reading order for data extraction)
        LinkedList<LinkedList<Rect>> ordered_text_zois = LexicographicAntichains(text_zois);

        CustomRegexToolbox rx = new CustomRegexToolbox();

        String symbole = new String();
        String serial = new String();
        String manufacturer = new String();
        String warranty = new String();
        String commission = new String();

        boolean symbole_found = false;
        boolean manufacturer_found = false;
        boolean warranty_found = false;
        boolean commission_found = false;

        LinkedList<Rect> serial_candidate_zois = new LinkedList<Rect>();

        // Loop over all text regions to extract symbole, manufacturer, warranty and commission
        // Serial id needs specific processing. It should not be extracted yet. Its anchor is
        // used to quote zones to be further analyzed.

        for (int l = 0; l < ordered_text_zois.size(); ++l)
        {
            LinkedList<Rect> text_line = ordered_text_zois.get(l);
            LinkedList<String> strings = new LinkedList<String>();
            int nb_zones = text_line.size();

            int symbole_search_pos = nb_zones;
            int manufacturer_search_pos = nb_zones;
            int warranty_search_pos = nb_zones;
            int commission_search_pos = nb_zones;

            boolean jump = false; // Skip anchor search for some regions

            // Step 1: OCR and anchor + data detection in each region

            for (int z = 0; z < nb_zones; ++z)
            {
                tess_engine.setRectangle(text_line.get(z));
                String str = tess_engine.getUTF8Text();
                String l_str = str.toLowerCase(); // Lower case to search anchors
                strings.add(str);
                Matcher anchor_matcher;

                //if (jump)
                //    jump = false;
                //else
                {
                    anchor_matcher = Pattern.compile(rx.getSerial_anchor()).matcher(l_str);
                    // Serial id processing postponed. Zones with such anchor are stored for
                    // specific analysis in further steps
                    if (anchor_matcher.find())
                        serial_candidate_zois.add(new Rect(text_line.get(z)));

                    anchor_matcher = Pattern.compile(rx.getSymbole_anchor()).matcher(l_str);

                    if (!symbole_found && anchor_matcher.find())
                    {
                        String sstr = str.substring(anchor_matcher.end());
                        Matcher mc = Pattern.compile(rx.getSymbole_base()).matcher(sstr);

                        if (mc.find()) {
                            symbole = sstr.substring(mc.start(), mc.end());
                            symbole_found = true;
                        }
                        else {
                            symbole_search_pos = z + 1;
                            jump = true;
                        }
                    }

                    anchor_matcher = Pattern.compile(rx.getManufacturer_anchor()).matcher(l_str);

                    if (!manufacturer_found && anchor_matcher.find())
                    {
                        String sstr = str.substring(anchor_matcher.end());
                        Matcher mc = Pattern.compile(rx.getManufacturer_base()).matcher(sstr);

                        if (mc.find()) {
                            manufacturer = sstr.substring(mc.start(), mc.end());
                            manufacturer_found = true;
                        }
                        else {
                            manufacturer_search_pos = z + 1;
                            jump = true;
                        }
                    }

                    anchor_matcher = Pattern.compile(rx.getWarranty_anchor()).matcher(l_str);

                    if (!warranty_found && anchor_matcher.find())
                    {
                        String sstr = str.substring(anchor_matcher.end());
                        Matcher mc = Pattern.compile(rx.getWarranty_base()).matcher(sstr);

                        if (mc.find()) {
                            warranty = sstr.substring(mc.start(), mc.end());
                            warranty_found = true;
                        }
                        else {
                            warranty_search_pos = z + 1;
                            jump = true;
                        }
                    }

                    anchor_matcher = Pattern.compile(rx.getCommission_anchor()).matcher(l_str);

                    if (!commission_found && anchor_matcher.find())
                    {
                        String sstr = str.substring(anchor_matcher.end());
                        Matcher mc = Pattern.compile(rx.getCommission_base()).matcher(sstr);

                        if (mc.find()) {
                            commission = sstr.substring(mc.start(), mc.end());
                            commission_found = true;
                        }
                        else {
                            commission_search_pos = z + 1;
                            jump = true;
                        }
                    }
                }
            }

            // Step 2: try to search data in zones right to anchor if not already found

            if (symbole_search_pos < nb_zones)
            {
                String str = strings.get(symbole_search_pos);
                Matcher mc = Pattern.compile(rx.getSymbole_base()).matcher(str);

                if (mc.find())
                {
                    symbole = str.substring(mc.start(), mc.end());
                    symbole_found = true;
                }
            }

            if (manufacturer_search_pos < nb_zones)
            {
                String str = strings.get(manufacturer_search_pos);
                Matcher mc = Pattern.compile(rx.getManufacturer_base()).matcher(str);

                if (mc.find())
                {
                    manufacturer = str.substring(mc.start(), mc.end());
                    manufacturer_found = true;
                }
            }

            if (warranty_search_pos < nb_zones)
            {
                String str = strings.get(warranty_search_pos);
                Matcher mc = Pattern.compile(rx.getWarranty_base()).matcher(str);

                if (mc.find())
                {
                    warranty = str.substring(mc.start(), mc.end());
                    warranty_found = true;
                }
            }

            if (commission_search_pos < nb_zones)
            {
                String str = strings.get(commission_search_pos);
                Matcher mc = Pattern.compile(rx.getCommission_base()).matcher(str);

                if (mc.find())
                {
                    commission = str.substring(mc.start(), mc.end());
                    commission_found = true;
                }
            }

            // Desperate attempt to extract date data without anchor

            for (int s = 0; s < nb_zones; ++s)
            {
                String str = strings.get(s);

                if (!commission_found)
                {
                    Matcher mc = Pattern.compile(rx.getCommission_base()).matcher(str);

                    if (mc.find())
                    {
                        commission = str.substring(mc.start(), mc.end());
                        commission_found = true;
                    }
                }
            }

        }

        tess_engine.end();

        //////////////////////////////////////////
        // Retrieve frame for serial identifier //
        //////////////////////////////////////////

        Mat cc_chart = new Mat(h, w, CvType.CV_8UC1);

        for (int k = 0; k < serial_candidate_zois.size(); ++k)
        {
            Rect rct = serial_candidate_zois.get(k);
            int top = rct.top;
            int bottom = rct.bottom;
            double distance_over = Double.POSITIVE_INFINITY;
            double distance_under = Double.POSITIVE_INFINITY;

            int id_ceil = 0;
            int id_floor = 0;

            for (int s = 0; s < solid_streams.size(); ++s)
            {
                Rect str = solid_streams.get(s);

                if (str.top < top)
                {
                    double distance = top - str.bottom;

                    if (distance < distance_over) {
                        distance_over = distance;
                        id_ceil = s;
                    }
                }
                else if (str.bottom > bottom)
                {
                    double distance = str.top - bottom;

                    if (distance < distance_under) {
                        distance_under = distance;
                        id_floor = s;
                    }
                }
            }

            Rect str_ceil = solid_streams.get(id_ceil);
            Rect str_floor = solid_streams.get(id_floor);
            Rect frm = new Rect(rct.left, str_ceil.top, rct.right, str_floor.bottom);

            // Lateral expansion for frame

            boolean expand = true;

            while (expand)
            {
                expand = false;

                int r = str_ceil.top;

                while (!expand && (r <= str_ceil.bottom))
                {
                    expand = (mask.get(r, frm.left)[0] == 0);
                    ++r;
                }

                r = str_floor.top;

                while (!expand && (r <= str_floor.bottom))
                {
                    expand = (mask.get(r, frm.left)[0] == 0);
                    ++r;
                }

                if (expand)
                    --frm.left;
            }

            frm.left = max(frm.left, 0);
            expand = true;

            while (expand)
            {
                expand = false;

                int r = str_ceil.top;

                while (!expand && (r <= str_ceil.bottom))
                {
                    expand = (mask.get(r, frm.right)[0] == 0);
                    ++r;
                }

                r = str_floor.top;

                while (!expand && (r <= str_floor.bottom))
                {
                    expand = (mask.get(r, frm.right)[0] == 0);
                    ++r;
                }

                if (expand)
                    ++frm.right;
            }

            frm.right = min(frm.right, w - 1);

            mat_bin.setTo(new Scalar(255));
            double[] ceil_local_thresholds = local_thresholds.get(id_ceil);
            double[] floor_local_thresholds = local_thresholds.get(id_floor);

            for (int c = frm.left; c <= frm.right; ++c)
            {
                final double th = (ceil_local_thresholds[c] + floor_local_thresholds[c]) / 2.0;

                for (int r = frm.top; r <= frm.bottom; ++r)
                    if (mat_lprime.get(r, c)[0] < th) mat_bin.put(r, c, 0);
                    else mat_bin.put(r, c, 255);
            }

            int max_cc_label = connectedComponents(mat_bin, cc_chart);

            Set<Integer> cc_touching_ceil = new HashSet<Integer>();
            Set<Integer> cc_touching_both = new HashSet<Integer>();

            for (int c = frm.left; c <= frm.right; ++c)
                cc_touching_ceil.add((int)cc_chart.get(frm.top, c)[0]);

            for (int c = frm.left; c <= frm.right; ++c)
            {
                int val = (int) cc_chart.get(frm.top, c)[0];

                if (cc_touching_ceil.contains(val))
                {
                    cc_touching_both.add(val);
                }
            }

            int nb_bars = cc_touching_both.size();

            Log.d(TAG, "bars : " + nb_bars);
        }



        System.out.println("symbol          : " + symbole);
        System.out.println("serial id       : " + serial);
        System.out.println("manufacturer id : " + manufacturer);
        System.out.println("warranty        : " + warranty);
        System.out.println("commission      : " + commission);



        Utils.matToBitmap(mat_bin, img_out);

        ((ImageView) findViewById(R.id.gray_display_view_name)).setImageBitmap(img_out);


        //img_gray.recycle(); // Sometimes make the app crash
        // Print text in IDE output console
        //System.out.println("OCR output over binary image:\n\n" + txt);

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

    private LinkedList<LinkedList<android.graphics.Rect>> LexicographicAntichains(LinkedList<android.graphics.Rect> rectangles)
    {
        int nb_rectangles = rectangles.size();
        LinkedList<LinkedList<android.graphics.Rect>> antichains = new LinkedList<LinkedList<android.graphics.Rect>>();
        LinkedList<android.graphics.Rect> residue_rcts = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> remaining_rcts = new LinkedList<android.graphics.Rect>(rectangles);

        // Search successive antichains of minimal rectangles for top-down order
        while (!remaining_rcts.isEmpty())
        {
            LinkedList<android.graphics.Rect> upper_rcts = new LinkedList<android.graphics.Rect>();

            for (int k = 0; k < remaining_rcts.size(); ++k)
            {
                android.graphics.Rect ref_rct = remaining_rcts.get(k);

                int kk = 0;
                boolean status = true;

                while (status && (kk < remaining_rcts.size()))
                {
                    if (kk != k)
                        status = !(ref_rct.top > remaining_rcts.get(kk).bottom);

                    ++kk;
                }

                if (status)
                    upper_rcts.add(new android.graphics.Rect(ref_rct));
                else
                    residue_rcts.add(new android.graphics.Rect(ref_rct));
            }

            Collections.sort(upper_rcts, new RectLeftComparator());
            antichains.add(upper_rcts);

            remaining_rcts.clear();
            remaining_rcts.addAll(residue_rcts);
            residue_rcts.clear();
        }

        return  antichains;
    }
}

class HorizontalSegmentOrdinateComparator implements Comparator<HorizontalSegment>{
    @Override
    public int compare(HorizontalSegment s1, HorizontalSegment s2) {
        return (int)(s1.ordinate - s2.ordinate);
    }
}

class RectTopComparator implements Comparator<android.graphics.Rect>{
    @Override
    public int compare(android.graphics.Rect r1, android.graphics.Rect r2) {
        return (int)(r1.top - r2.top);
    }
}

class RectLeftComparator implements Comparator<android.graphics.Rect>{
    @Override
    public int compare(android.graphics.Rect r1, android.graphics.Rect r2) {
        return (int)(r1.left - r2.left);
    }
}

