import React from 'react';
import './Feature.css';
import { useHistory } from "react-router-dom";
import { Button } from 'react-bootstrap';


function Feature(props) {
    const history = useHistory();

  function handleClick() {
    history.push('/ScanPost')
  }
    return (
      <div>
          <button class="ui blue button" className="btn1">{props.text}</button>
            {/* <button type="button" class="btn btn-primary btn-lg" onClick={handleClick}>
                {porps.text}
            </button> */}
        </div>
    );
}

export default Feature;