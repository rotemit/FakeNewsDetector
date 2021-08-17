import React, { useState, useEffect } from 'react';
// import axios from "axios";

import { Posts } from '../Components/Posts/Posts';
// import { useHistory } from "react-router-dom";
// import {Posts} from "./../Components/Posts/Posts";
import './EnterProfile.css'

export const EnterProfile = () => {
  const [name, setName] = useState('');
  const [posts, setPost] = useState([]);
  // let counter = 1;

  // useEffect(() => {
  //   axios.get('/profile').then((response) => {
  //     setPost(response.name);
  //   });
  // }, []);

  useEffect(() => {
    
    fetch('/profile').then(response =>  
      response.json().then(data => {
      setPost(data.name);
      console.log(data.name);
    })
    )
    
    //  setPost(response);
  }, []);

  

 return(
  <div className='form'>
    <h1 className='h1'>Scan profile</h1>
    <form>
        <label>
          <input className='input' 
          value={name} 
          placeholder="Enter Url"
          onChange={e => setName(e.target.value)} />
        </label>
      
      <button className='btn' classtype="submit" onClick={async () => {
        
        const response = await fetch('/profile', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(name)
        });
        if (response.ok) {

          console.log('response worked!');
        }
      }}>
        Submit
      </button>
      </form>  
      <div>
        <Posts className='list' posts={posts}/>      
      </div>
   
  </div>
  
)

 
 
}

