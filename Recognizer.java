import android.content.ContentResolver;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import java.io.FileDescriptor;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Recognition {

      private final String id;
      private final String title;
      private final Float confidence;
      private RectF location;

      Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }
}


public class Recognizer {

    private static Recognizer recognizer;

    private YOLO yolo;
    private Embedding embedding;
    private LibSVM svm;

    private List<String> classNames;

    private Recognizer() {}

    static Recognizer getInstance (AssetManager assetManager) throws Exception {
        if (recognizer != null) return recognizer;

        recognizer = new Recognizer();
        recognizer.yolo = YOLO.create(assetManager);
        recognizer.embedding = Embedding.create(assetManager);
        recognizer.svm = LibSVM.getInstance();
        recognizer.classNames = FileUtils.readLabel(FileUtils.LABEL_FILE);

        return recognizer;
    }

    CharSequence[] getClassNames() {
        CharSequence[] cs = new CharSequence[classNames.size()-4];
        int idx = 1, i=0;

        cs[0] = "new entry";
        for (String name : classNames) {
            if(i>4)
                cs[idx++] = name;
            i++;
        }

        return cs;
    }

    List<Recognition> recognizeImage(Bitmap bitmap, Matrix matrix) {
        synchronized (this) {
            List<RectF> faces = yolo.detect(bitmap);
            final List<Recognition> mappedRecognitions = new LinkedList<>();

            for (RectF rectF : faces) {
                Rect rect = new Rect();
                rectF.round(rect);

                FloatBuffer buffer = embedding.getEmbeddings(bitmap, rect);
                LibSVM.Prediction prediction = svm.predict(buffer);

                matrix.mapRect(rectF);
                int index = prediction.getIndex();

                String name = classNames.get(index);
                Recognition result =
                        new Recognition("" + index, name, prediction.getProb(), rectF);
                mappedRecognitions.add(result);
            }
            return mappedRecognitions;
        }

    }

    void updateData(int label, ContentResolver contentResolver, ArrayList<Uri> uris) throws Exception {
        synchronized (this) {
            ArrayList<float[]> list = new ArrayList<>();

            for (Uri uri : uris) {
                Bitmap bitmap = getBitmapFromUri(contentResolver, uri);
                List<RectF> faces = yolo.detect(bitmap);

                Rect rect = new Rect();
                if (!faces.isEmpty()) {
                    faces.get(0).round(rect);
                }

                float[] emb_array = new float[Embedding.EMBEDDING_SIZE];
                embedding.getEmbeddings(bitmap, rect).get(emb_array);
                list.add(emb_array);
            }

            svm.train(label, list);
        }
    }

    int addPerson(String name) {
        FileUtils.appendText(name, FileUtils.LABEL_FILE);
        classNames.add(name);

        return classNames.size();
    }

    private Bitmap getBitmapFromUri(ContentResolver contentResolver, Uri uri) throws Exception {
        ParcelFileDescriptor parcelFileDescriptor =
                contentResolver.openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();

        return bitmap;
    }

    void close() {
        yolo.close();
        embedding.close();
    }
}
