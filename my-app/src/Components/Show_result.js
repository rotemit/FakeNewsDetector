// import React from 'react';
// import './Posts.css';

export const Show_result = (props) => {
    
    if (props.result === 0.0) {
        return <div>no result</div>
    } else {
        return <div>{props.result}</div>
    }

}