import React from "react";
import "./gallery.css";

function Gallery({ imageData, resultPage, showDesc, loadMore }) {
  const imagesPerPage = 4;
  return (
    <>
      <div className="skills__container container grid">
        {imageData &&
          imageData.slice(0, imagesPerPage * resultPage).map((img) => (
            <div key={img.img_rank}>
              <img
                src={"data:image/png;base64," + img.img_data}
                alt="marble_image"
              />
              {showDesc && <div className="desc">{img.img_desc}</div>}
            </div>
          ))}
      </div>

      {imageData && resultPage < imagesPerPage && (
        <div className="load-more-btn">
          <button onClick={() => loadMore()}>Load More</button>
        </div>
      )}
    </>
  );
}

export default Gallery;
