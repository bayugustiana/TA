<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Gateway extends Model
{
    // Table Name
    protected $table = 'gateway';
    // Increment Setting
    public $incrementing = false;
    // Primary Key
    public $primary_key = 'qrcode';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_tempat',
        'status'
    ];

    public function tempat()
    {
        return $this->belongsTo('App\Models\Tempat','id_tempat');
    }

}
