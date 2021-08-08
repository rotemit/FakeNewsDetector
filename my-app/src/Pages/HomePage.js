import React from 'react';
// import { Form } from 'semantic-ui-react';
import Feature from '../Components/Features/Feature';
import './HomePage.css'
export const HomePage = () => {
    // const {todo, setTodo} = useState([])
   
    return(
        <div className='form'>
            <Feature text="Scan Profile"/>
            <Feature text="Scan post"/>
            <Feature text="about"/>
            <Feature text="contact"/>
        </div>    
    )
}