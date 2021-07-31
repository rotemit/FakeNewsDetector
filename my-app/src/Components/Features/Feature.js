import React from 'react';
import './Feature.css';
import { useHistory } from 'react-router-dom';

function Feature(porps) {
    const history = useHistory();
    return (
      <div>
          <div className='actions'>
            <button className='btn' onClick={() => history.push('../../Pages/HomePage.js')}>
                {porps.text}
            </button>
        </div>
      </div>
    );
}

export default Feature;