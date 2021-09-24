import React from 'react';
import './Feature.css';
import { useHistory } from "react-router-dom";

function Feature(porps) {
    const history = useHistory();

  function handleClick() {
    history.push('/ScanPost')
  }
    return (
      <div>
          <div className='actions'>
            <button className='btn' onClick={handleClick}>
                {porps.text}
            </button>
        </div>
      </div>
    );
}

export default Feature;