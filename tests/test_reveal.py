import unittest
from PIL import Image
from pi_sensenova.live import reveal


class RevealTests(unittest.TestCase):
    def test_blank_to_exact_final_without_mutation(self):
        image = Image.new("RGB", (32, 32), "white")
        original = image.tobytes()
        values = [reveal(image, n, 10).getpixel((0, 0))[0] for n in (0, 3, 6, 9, 10)]
        self.assertEqual(values[0], 128)
        self.assertEqual(values[-1], 255)
        self.assertEqual(values, sorted(values))
        self.assertIs(reveal(image, 10, 10), image)
        self.assertEqual(image.tobytes(), original)
