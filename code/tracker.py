import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ParticleFilter import ParticleFilter


class Tracker():
    def __init__(self, video, player_detections, line_annotations):
        self.video = cv2.VideoCapture(video)
        self.detection_file = player_detections
        self.line_file = line_annotations

        self.PHash = cv2.img_hash_PHash().create()
        self.old_hash = None
        self.pitch = cv2.resize(cv2.imread('pitch_image.png'), None, fx=0.25, fy=0.25)
        self.counter = 0
        self.H = None

        self.filters = []

    def _get_lines(self, frame):
        hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
        vals = np.histogram(hue, bins=180)[0]  # Find most common hue value

        pitch_mask = np.ones_like(hue) * 255
        pitch_mask[hue >= np.argmax(vals) + 5] = 0
        pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_ERODE, np.ones((20, 20)))  # Convert to mask
        pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, np.ones((50, 50))) <= 25  # Convert to mask

        l = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)[:, :, 0]
        bg = cv2.medianBlur(l, 15)

        edges = l - bg
        edges[edges < 10] = 255
        edges[pitch_mask] = 255
        edges = (edges < 128).astype(float)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5)))

        edges = cv2.Canny(edges.astype(np.uint8), 50, 100, apertureSize=7)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9)))
        lines = cv2.HoughLinesP(edges, 3, np.pi / 180, 800)

        final = np.zeros((720, 1280))
        if lines is not None:
            for [[x1, y1, x2, y2]] in lines:
                if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 > 100:
                    cv2.line(final, (x1, y1), (x2, y2), 255, 5)

        final[pitch_mask] = 0
        return final, lines

    def _get_equations(self, lines):
        if lines is None:
            return None
        out = []
        for [[x1, y1, x2, y2]] in lines:
            # Filter out any that are too short
            if ((x1 - x2) ** 2 + (y1 + y2) ** 2) ** 0.5 < 10:
                continue

            # Convert to the form ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1

            # Filter out any duplicates
            use = True
            for ap, bp, cp, _, _, _, _ in out:
                # First catches near to 0, second is larger values
                if (np.isclose(a, ap) or np.isclose(a / ap, 1, 1)) \
                        and (np.isclose(b, bp) or np.isclose(b / bp, 1, 1)) \
                        and (np.isclose(c, cp) or np.isclose(c / cp, 1, 1)):
                    use = False
                    break
            if use:
                out.append((a, b, c, x1, y1, x2, y2))
        if len(out) > 0:
            return np.vstack(out)

    def _find_intersections(self, lines, is_footage):
        intersections = []
        for i in range(len(lines)):
            for j in range(i, len(lines)):
                if i == j:
                    continue

                if is_footage:
                    t1, a1, b1, c1, _, _, _, _ = lines[i]
                    t2, a2, b2, c2, _, _, _, _ = lines[j]
                else:
                    try:
                        t1, a1, b1, c1 = lines[i]
                        t2, a2, b2, c2 = lines[j]
                    except TypeError:
                        return # If not enough lines to iterate over

                if t1 == t2 or t1 == -1 or t2 == -1:
                    # Filter out as they're the same line, or they're invalid
                    continue

                try:
                    x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1), (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
                except ZeroDivisionError:
                    # Likely the same line
                    continue

                    #             # Pitch bounds fall within this too - we know they will be correct.
                    #             if x[0] < 0 or x[0] > 1280 or x[1] < 0 or x[1] > 720:
                    #                 continue
                x = np.concatenate((np.array([min(t1, t2), max(t1, t2)]), x))
                intersections.append((x))
        if intersections:
            return np.vstack(intersections)
        return None

    def _draw_points(self, frame, intersections):
        for [_, _, x, y] in intersections:
            cv2.circle(frame, (int(x), int(y)), 15, (255, 0, 0), -1)
        return frame

    def _draw_lines(self, frame, equations, oned=True):
        if equations is None:
            return frame
        for a, b, c, _, _, _, _ in equations:
            try:
                p1 = 0, int(-c / b)
                p2 = 1280, int(-((c + 1280 * a) / b))
                if oned:
                    cv2.line(frame, p1, p2, 128, 2)
                else:
                    cv2.line(frame, p1, p2, (0, 255, 0), 4)
            except:
                continue
        return frame

    def _get_pitch_equation(self, line):
        # Useful for debugging purposes
        values = {
            0: 'left_try',
            1: 'left_five',
            2: 'left_twenty_two',
            3: 'left_ten',
            4: 'halfway',
            5: 'right_ten',
            6: 'right_twenty_two',
            7: 'right_five',
            8: 'right_try',
            9: 'top_touch',
            10: 'top_5',
            11: 'top_15',
            12: 'bottom_15',
            13: 'bottom_5',
            14: 'bottom_touch',
        }

        if line == 0:
            return (0, 1, 0, -83)
        elif line == 1:
            return (1, 1, 0, -104)
        elif line == 2:
            return (2, 1, 0, -178)
        elif line == 3:
            return (3, 1, 0, -257)
        elif line == 4:
            return (4, 1, 0, -300)
        elif line == 5:
            return (5, 1, 0, -343)
        elif line == 6:
            return (6, 1, 0, -421)
        elif line == 7:
            return (7, 1, 0, -495)
        elif line == 8:
            return (8, 1, 0, -517)
        elif line == 9:
            return (9, 0, 1, -14)
        elif line == 10:
            return (10, 0, 1, -34)
        elif line == 11:
            return (11, 0, 1, -78)
        elif line == 12:
            return (12, 0, 1, -252)
        elif line == 13:
            return (13, 0, 1, -295)
        elif line == 14:
            return (14, 0, 1, -317)

    def _get_point_pairs(self, footage_points):
        out = []
        if footage_points is not None:
            for t1, t2, x, y in footage_points:
                res = self._find_intersections([self._get_pitch_equation(t1), self._get_pitch_equation(t2)], False)
                if res is None:
                    continue
                out.append((x, y, res[0, 2], res[0, 3]))
        if len(out) >= 4:
            return True, np.vstack(out)
        return False, None

    def _get_detection_positions(self, detections):
        out = np.zeros((len(detections), 2))
        for i, (x1, y1, x2, y2) in enumerate(detections):
            out[i] = (x1+x2)/2, (y1+y2)/2
        return out.astype(np.float32)

    def _translate_points(self, footage_points):
        points = np.zeros((len(footage_points), 2), np.float32)
        for i, r in enumerate(np.insert(footage_points, 2, 1, 1)):
            pprime = self.H @ r
            pprime = pprime[0] / pprime[2], pprime[1] / pprime[2]
            points[i] = pprime
        return points

    def _update_filters(self, changed, points):
        if changed:
            # Changed scene, so reinitialise the kalman filters
            self.filters = []

        if not self.filters:
            for point in points:
                self.filters.append(ParticleFilter(500, point))
            return points
        else:
            dists = np.empty((len(self.filters), len(points)))
            for i, filter in enumerate(self.filters):
                filter.predict()
                dists[i] = np.linalg.norm(points - filter.estimate(), axis=1)

            new_filters = []
            try:
                row, col = linear_sum_assignment(dists)
            except ValueError:
                # linear_sum can fail at times, we revert to old filters if so
                return self.filters
            for i in range(len(col)):
                # Row filters, col points
                if dists[row[i], col[i]] > 25:
                    pass
                    # Too far - new pf from this
                    new_filters.append(ParticleFilter(500, points[col[i]]))
                else:
                    filter = self.filters[row[i]]
                    # Add the old filter to the new list
                    new_filters.append(filter)
                    filter.update(points[col[i]])
                    filter.resample()

            # Let any non-updated filters decay
            for filter in self.filters:
                # Inefficient but works
                if filter not in new_filters and not filter.update_none():
                    # Gets to live another day
                    new_filters.append(filter)

            # Update to new filters
            self.filters = new_filters

    def get_frame(self, add_lines, add_detections, add_translations, add_particles):
        ok, frame = self.video.read()

        # Check if the frame has changed
        new_hash = self.PHash.compute(frame)
        changed_scene = False
        if self.old_hash is not None:
            if self.PHash.compare(self.old_hash, new_hash) > 10:
                # We've changed scene
                changed_scene = True
        self.old_hash = new_hash

        tmp = self.pitch.copy()
        out, lines = self._get_lines(frame)
        #         f[out > 10] = 255

        lines = self._get_equations(lines)
        if add_lines:
            frame = self._draw_lines(frame, lines)

        # We load in the labels here, they are simply an extra column prefixed with the correct labelling
        lines = np.array(self.line_file['frame' + str(self.counter)])

        if lines is not None:
            intersections = self._find_intersections(lines, True)

            valid, pairs = self._get_point_pairs(intersections)
            if valid:
                Hn, msk = cv2.findHomography(pairs[:, :2], pairs[:, 2:])
                # If we have a new homography, use it, otherwise we use the old value
                if Hn is not None:
                    self.H = Hn

            dets = np.array(self.detection_file['frame' + str(self.counter)])
            player_points = self._get_detection_positions(dets)

            if add_detections:
                for i in range(len(dets)):
                    cv2.rectangle(frame, (dets[i, 0], dets[i, 1]), (dets[i, 2], dets[i, 3]), (0, 0, 255), 1)

            if self.H is not None:
                # Might not be able to translate if we don't know H
                player_points = self._translate_points(player_points)
                self._update_filters(changed_scene, player_points)

            # Draw the overlays
            if add_particles:
                for filter in self.filters:
                    filter.draw(tmp)

            if add_translations:
                for x, y in player_points:
                    try:
                        cv2.circle(tmp, (x, y), 3, (255, 0, 255), -1)
                    except ValueError:
                        # May be NaN, continue if so
                        continue

        self.counter += 1
        return tmp, frame, changed_scene, self.H, len(self.filters), self.counter

