package com.example.jeanduong.framedplateapplication;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static java.lang.Math.max;
import static java.lang.Math.min;

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
        // Setup
        //
        // Bitmap is only needed to load image from ressources and create data array.
        // It can be deleted after that, since only Mat objects will be used for processing

        Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.balser);

        final int h = img_rgb.getHeight();
        final int w = img_rgb.getWidth();
        Mat mat_rgb = new Mat(h, w, CvType.CV_8UC3);

        Utils.bitmapToMat(img_rgb, mat_rgb);
        img_rgb.recycle(); // May make the app crash

        //Imgproc.GaussianBlur(mat_rgb, mat_rgb, new Size(5, 5), 0.0); // Blur to remove some noise

        // Leydier's pseudo-luminance

        Imgproc.cvtColor(mat_rgb, mat_rgb, Imgproc.COLOR_BGR2RGB); // OpenCV color image are BGR!!!
        Mat mat_lprime = MakeLPrime(mat_rgb);
        mat_rgb.release(); // May make the app crash

        //Imgproc.GaussianBlur(mat_lprime, mat_lprime, new Size(5, 5), 0.0); // Blur to remove some noise

        // Canny edge detector (keep only sites with values "around average luminance")

        Mat mat_edges = new Mat(h, w, CvType.CV_8UC1);
        
        MatOfDouble mu = new MatOfDouble();
        MatOfDouble sigma = new MatOfDouble();
        Core.meanStdDev(mat_lprime, mu, sigma);

        double m = mu.get(0, 0)[0];
        double s = sigma.get(0, 0)[0];
        double th_low = m - s;
        double th_high = m + s;

        Imgproc.Canny(mat_lprime, mat_edges, th_low, th_high);

        // Otsu binarization

        Mat mat_bin = new Mat(h, w, CvType.CV_8UC1);

        Imgproc.threshold(mat_lprime, mat_bin, 0.0, 255, Imgproc.THRESH_OTSU);

        Core.bitwise_not(mat_bin, mat_bin);

        // Distance transform

        Mat mat_dst = new Mat(h, w, CvType.CV_8UC1);

        Imgproc.distanceTransform(mat_bin, mat_dst, Imgproc.CV_DIST_L2, 3);

        Core.MinMaxLocResult mm = Core.minMaxLoc(mat_dst);

        int min_val = (int) mm.minVal;
        int max_val = (int) mm.maxVal;

        Log.i(TAG, "Minimal distance = " + min_val);
        Log.i(TAG, "Maximal distance = " + max_val);

        /*
        // Threshold laplacian magnitudes using pseudo-luminance as weights

        // Very poor results!!!

        Mat mat_Laplacian = new Mat(h, w, CvType.CV_16SC1);

        Imgproc.Laplacian(mat_lprime, mat_Laplacian, CvType.CV_16S);

        double max_gray = 0.0;

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                max_gray = max(max_gray, mat_lprime.get(r, c)[0]);

        double cumul_bright = 0.0, cumul_dark = 0.0;
        double count_bright = 0.0, count_dark = 0.0;

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
            {
                double value = mat_Laplacian.get(r, c)[0];
                double w_bright = mat_lprime.get(r, c)[0];
                double w_dark = max_gray - w_bright;

                cumul_bright += value * w_bright;
                cumul_dark += value * w_dark;
                count_bright += w_bright;
                count_dark += w_dark;
            }

        double th = (cumul_bright / count_bright + cumul_dark / count_dark) / 2.0;

        Mat mat_bin = new Mat(h, w, CvType.CV_8UC1);

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
            {
                if (mat_Laplacian.get(r, c)[0] < th)
                    mat_bin.put(r, c, 0);
                else
                    mat_bin.put(r, c, 255);
            }
        */

        // MSER detection
/*
        MatOfKeyPoint mokp = new MatOfKeyPoint();
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.MSER);

        detector.detect(mat_lprime, mokp);

        Log.i(TAG, "|MSER| = " + mokp.rows());

        Features2d.drawKeypoints(mat_lprime, mokp, mat_lprime, new Scalar(255), Features2d.DRAW_RICH_KEYPOINTS);
*/
        // Display new image
        Bitmap img_gray = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_lprime, img_gray);
        //Utils.matToBitmap(mat_bin, img_gray);
        //Utils.matToBitmap(mat_edges, img_gray);
        ((ImageView) findViewById(R.id.gray_display_view_name)).setImageBitmap(img_gray);

        mat_lprime.release(); // May make the app crash

        tess_engine = new TessBaseAPI();
        datapath = getFilesDir() + "/tesseract/";

        //make sure training data has been copied
        //checkFile(new File(datapath + "tessdata/"));

        tess_engine.init(datapath, lang);
        tess_engine.setImage(img_gray);

        String txt = tess_engine.getUTF8Text();

        tess_engine.end();



        //img_gray.recycle(); // Sometimes make the app crash
        // Print text in IDE output console
        Log.i(TAG, txt);
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

/*
        // Memory consuming version


        Log.i(TAG, mat_rgb.channels() + " channels");

        byte[] flatten_data = new byte[(int) mat_rgb.total() * 3];
        mat_rgb.get(0, 0, flatten_data);
        int k = -1;

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
        {
            ++k;
            red = flatten_data[k];
            ++k;
            green = flatten_data[k];
            ++k;
            blue = flatten_data[k];

            ma = max(red, max(green, blue));
            mi = min(red, min(green, blue));

            if (ma > 0.0)
                s = 255.0 * (1.0 - (mi / ma));
            else
                s = 0.0;

            v = ((255.0 - s) * (red * 0.299 + green * 0.587 + blue * 0.114)) / 255.0;

            mat_L_prime.put(r, c, (int)min(255.0, max(0.0, v)));
        }
*/

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

}
