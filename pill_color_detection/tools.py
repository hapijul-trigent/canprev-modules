import webcolors

webcolors._definitions._CSS3_HEX_TO_NAMES.update(CUSTOM_CSS3_COLORS_HEX_MAP)


# Fetch Color HEX
def get_color_name_by_rgb(rgb):
    """
    Convert RGB to the closest known web color name.

    :param rgb: Tuple containing Red, Green, Blue (R, G, B) values (each 0-255)
    :return: Closest color name, exact color name if found, or 'Unknown'
    """
    try:
        # Get the exact color name if it matches
        hex_color = webcolors.rgb_to_hex(rgb)
        return hex_color, webcolors.hex_to_name(hex_color)
    except ValueError:
        # If no exact match is found, find the closest color name
        closest_color = min(webcolors._definitions._CSS3_HEX_TO_NAMES.items(),
                            key=lambda x: sum((v - c) ** 2 for v, c in zip(webcolors.hex_to_rgb(x[0]), rgb)))
        return closest_color


# Example usage:
if __name__ == '__main__':
    rgb_color = (205, 133, 69)
    color_name = get_color_name_by_rgb(rgb_color)
    print(f"RGB {rgb_color} -> Closest color name: {color_name}")