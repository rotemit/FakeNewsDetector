// import React from 'react';
import "./ScanPost.css";
import React, { useState, useEffect } from 'react';

// import { Show_result } from "../Components/Show_result";

export const ScanPost = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState({trust:'', machine:'', semantic:''});
  const [isClicked, setIsClicked] = useState(false);

  const onClick = (e) => {
    e.preventDefault();
    console.log('You clicked submit.');
    console.log(result.trust);
    setIsClicked(true);
  };

  function Result() {
    return (
      <div>
        <p>Trust result: {result.trust}</p>
        <p>Machine result: {result.machine}</p>
        <p>Semantic result: {result.semantic}</p>
        <h2>Total: </h2>
      </div>
    )
  }

  async function handleSubmit()  {   
      const response = await fetch('/scanPost', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(text)
      })
      if (response.ok) {
        console.log('response worked!');
      } else {

      }
  }
  

    useEffect(() => {
      fetch('/scanPost').then(response => response.json()).then(data => {
        setResult({trust: data.trust_value, machine: data.machine_value, semantic: data.semantic_value});
      });
      
            
    }, []);

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
          
          <button className='btn' classtype="submit" onClick={handleSubmit}>
            Submit
          </button>
          <div>
          <button className='btn' onClick={onClick} > 
            Show results
          </button>
          
          </div>
          </form>    
          {(isClicked)? <Result /> : <p>No result</p>     }
      </div>


    )
}