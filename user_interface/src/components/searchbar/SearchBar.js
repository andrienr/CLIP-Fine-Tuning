import React from "react";
import Switch from "@mui/material/Switch";
import FormControlLabel from "@mui/material/FormControlLabel";
import "./searchbar.css";

function SearchBar({
  handleSubmit,
  queryInput,
  handleQueryInput,
  handleShowDesc,
}) {
  return (
    <div className="search-bar__container container">
      <form
        className="search-bar__form"
        onSubmit={
          queryInput
            ? handleSubmit
            : () => setBlank_field_msg("Please fill the search box!")
        }
      >
        <FormControlLabel
          control={<Switch onClick={() => handleShowDesc()} />}
          label="Show description"
        />
        <input
          className="search-bar__query-input"
          name="query"
          value={queryInput}
          type="text"
          id="search"
          placeholder="Search for Images"
          onChange={(e) => handleQueryInput(e)}
          required
        />
        <input
          type="submit"
          value="Search"
          id="submit"
          className="search-bar__search-button"
        />
      </form>
    </div>
  );
}

export default SearchBar;
