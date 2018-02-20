    _, cnts, _ = cv.findContours(
        mask.copy(),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if len(cnts):
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        for c in cnts:
            if cv.contourArea(c) > 10000:
                # compute the rotated bounding box of the contour
                # orig = frame.copy()
                box = cv.minAreaRect(c)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                cv.drawContours(frame, [box.astype("int")], -1, RGB.Cyan, 2)

                for (x, y) in box:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the
                # top-left and top-right points
                # followed by the midpoint between the
                # top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                cv.circle(frame, (int(tltrX), int(tltrY)), 5, RGB.Red, -1)
                cv.circle(frame, (int(blbrX), int(blbrY)), 5, RGB.Red, -1)
                cv.circle(frame, (int(tlblX), int(tlblY)), 5, RGB.Red, -1)
                cv.circle(frame, (int(trbrX), int(trbrY)), 5, RGB.Red, -1)

                # draw lines between the midpoints
                cv.line(frame, (int(tltrX), int(tltrY)),
                        (int(blbrX), int(blbrY)),
                        RGB.Magenta, 2)
                cv.line(frame, (int(tlblX), int(tlblY)),
                        (int(trbrX), int(trbrY)),
                        RGB.Magenta, 2)

                # compute the Euclidean distance between the midpoints
                dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # compute the size of the object
                dimA = dA / pixelsPerMetricX
                dimB = dB / pixelsPerMetricY

                # draw the object sizes on the image
                cv.putText(frame, "{:.1f} mm".format(dimA),
                           (int(tltrX - 15), int(tltrY - 10)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           2, RGB.Lime, 2)
                cv.putText(frame, "{:.1f} mm".format(dimB),
                           (int(trbrX + 10), int(trbrY)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           2, RGB.Lime, 2)
                