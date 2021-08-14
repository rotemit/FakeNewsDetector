import React from 'react';
import './Posts.css';

export const Posts = ({posts}) => {
    console.log(posts)
    // const itemsArr = posts.split(',');
//     const listItems = a.map((post) =>
//     <li>{post}</li>
//   );
//   return (
//     <ul>{listItems}</ul>
//   );
    return (
        <ul className='list'>
            {posts.map((post) => <li className='post' key={post}> {post}</li>)}
        </ul>
    )
}