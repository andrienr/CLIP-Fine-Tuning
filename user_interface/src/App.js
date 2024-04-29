import { useState } from "react";
import "./app.css";
import axios from "axios";
import Title from "./components/title/Title";
import SearchBar from "./components/searchbar/SearchBar";
import Gallery from "./components/gallery/Gallery";

export default function App() {
  const [imageData, setImageData] = useState(null);
  const [resultPage, setResultpage] = useState(1);
  const [queryInput, setQueryInput] = useState("");
  const [showDesc, setShowDesc] = useState(false);

  function handleQueryInput(e) {
    setQueryInput(e.target.value);
  }

  function handleShowDesc() {
    setShowDesc(!showDesc);
  }

  function handleSubmit(e) {
    e.preventDefault();
    setResultpage(1);
    axios
      .get(`/marbles/${queryInput}`)
      .then(function (response) {
        setImageData(response.data);
      })
      .catch(function (error) {
        console.log(error);
      });
  }

  function loadMore() {
    setResultpage(resultPage + 1);
  }

  return (
    <>
      <Title />
      <SearchBar
        handleSubmit={handleSubmit}
        handleQueryInput={handleQueryInput}
        queryInput={queryInput}
        handleShowDesc={handleShowDesc}
      />
      <Gallery
        imageData={imageData}
        resultPage={resultPage}
        showDesc={showDesc}
        loadMore={loadMore}
      />
    </>
  );
}
