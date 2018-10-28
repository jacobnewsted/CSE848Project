using System;
using System.Collections.Generic;
using System.Text;

namespace GANLibrary
{
    public class Utilities
    {
        private Utilities() { }

        public static int[,] StringsToVector(string[] image)
        {
            int[,] vectorList = new int[image.Length, image[0].Length];

            for (int i = 0; i < image.Length; i++)
            {
                for (int j = 0; j < image[i].Length; j++)
                {
                    vectorList[i, j] = Convert.ToInt32(image[i][j]);
                }
            }

            return vectorList;
        }

        public static string[] VectorToStrings(int[,] image)
        {
            string[] toReturn = new string[image.Length];

            for (int i = 0; i < image.GetLength(0); i++)
            {
                string build = "";
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    build += Convert.ToString(image[i, j]);
                }
                toReturn[i] = build;
            }

            return toReturn;
        }
    }
}
