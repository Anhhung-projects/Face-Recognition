img_path = os.path.join(path, 'Testing', 'female_1.jpg')
result_img = GenderRecognition(img_path)
# result_img = cv2.imread('Testing\image_1.jpg')
if result_img is not None:
    cv2.imwrite('Face Detection', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('DONE')
else:
    print("Error: No image to display!")