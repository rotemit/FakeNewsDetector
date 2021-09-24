// import React from 'react';
import "./ScanPost.css";
import React, { useState, useEffect } from 'react';

export const ScanPost = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState(0.0);
    
    function onClick() {
        console.log(result);
    }

    useEffect(() => {
          fetch('/scanPost').then(response =>  
            response.json().then(data => {
            setResult(data.name);
            console.log(data.name);
            })
            )
        
    });
    return (
        <div className='form'>
        <h1 className='h1'>Scan post</h1>
        <form>
            <label>
              <input className='input' 
              value={text} 
              placeholder="Enter text"
              onChange={e => setText(e.target.value)} />
            </label>
          
          <button className='btn' classtype="submit" onClick={async () => {
            
            const response = await fetch('/scanPost', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify(text)
            })
            if (response.ok) {
              // setIsDone(true)
              console.log('response worked!');
            }
          }}>
            Submit
          </button>
          <div>
          <button className='btn' onClick={onClick} > 
            Show results
          </button>
          </div>
         
          </form>  
       
      </div>


    )
}