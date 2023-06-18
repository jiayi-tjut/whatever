void CMFCApplication1View::erosion(BYTE* image, int w, int h, BYTE* outImg)
{
	int rept;
	//腐蚀
	memcpy(outImg, image, sizeof(BYTE) * width * height);
	int i, j;
	int m, n;
	BYTE flag;
	for (rept = 0; rept < 3; rept++)
	{
		for (i = 0; i < h-1; i++)
		{
			for (j = 0; j < w - 1; j++)
			{
				if (image[i * w + j] == 255)
				{
					flag = 0;
					for (m = -1; m < 2; m++)
					{
						for (n = -1; n < 2; n++)
						{
							if (image[(i + m) * 3 + j + n] == 0)
							{
								flag++;
							}
						}
					}
					if (flag > 2)
					{
						image[(i * m + j)] = 0;
					}
				}

			}
		}
		memcpy(image, outImg, sizeof(BYTE) * width * height);
	}
}
