using System;
using System.IO;

namespace NeuroConsole {
    public class MNISTReader {
        /// <summary>
        /// Read an MNIST data file at the specified
        /// location.
        /// </summary>
        /// <param name="size">
        /// The size of the MNIST set.
        /// </param>
        /// <param name="labelLocation">
        /// The location of the MNIST file.
        /// </param>
        /// <returns>
        /// an array of bytes containing the numbers which
        /// the written images have been classified as.
        /// </returns>
        public static byte[] ReadMNISTLabels(int size, string labelLocation)
        {
            byte[] labels = null;
            FileStream fs = null;
            
            try
            {
                labels = new byte[size];
                fs = new FileStream(labelLocation, FileMode.Open);
                using (BinaryReader labelBinaryReader = new BinaryReader(fs))
                {
                    // Read a padding byte
                    labelBinaryReader.ReadInt32(); 
                    int numLabels = labelBinaryReader.ReadInt32();

                    for (int i = 0; i < size; i++)
                        labels[i] = labelBinaryReader.ReadByte();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                fs.Dispose();
            }
            
            return labels;
        }

        /// <summary>
        /// Read an MNIST data file at the specified
        /// location.
        /// </summary>
        /// <param name="size">
        /// The size of the MNIST data set.
        /// </param>
        /// <param name="imageSize">
        /// The size of the images in the dataset. Since the
        /// images can only be square, you can only specify
        /// one number.
        /// </param>
        /// <param name="imageLocation"></param>
        /// <returns></returns>
        public static byte[][][] ReadMNISTImages(int size, int imageSize, string imageLocation)
        {
            // Initialize the array of images with an array
            // of images[size][imageSize][imageSize]
            byte[][][] images = new byte[size][][];
            for (int i = 0; i < images.Length; i++) {
                images[i] = new byte[imageSize][];

                for (int j = 0; j < images[i].Length; j++)
                    images[i][j] = new byte[imageSize];
            }

            FileStream fs = null;

            try
            {
                fs = new FileStream(imageLocation, FileMode.Open); 
                using (BinaryReader imageBinaryReader = new BinaryReader(fs))
                {
                    // Read a padding byte
                    imageBinaryReader.ReadInt32();
                    int numImages = imageBinaryReader.ReadInt32(); 
                    int numRows = imageBinaryReader.ReadInt32(); 
                    int numCols = imageBinaryReader.ReadInt32();

                    for (int s = 0; s < size; s++)
                        for (int i = 0; i < imageSize; i++)
                            for (int j = 0; j < imageSize; j++) {
                                byte b = imageBinaryReader.ReadByte();
                                images[s][i][j] = b;
                            }

                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                fs.Dispose();
            }

            return images;
        }
    }
}
